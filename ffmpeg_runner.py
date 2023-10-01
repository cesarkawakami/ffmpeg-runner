import glob
import json
import os.path
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Optional, Set


def get_input_duration_secs(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        "-i",
        input_path,
    ]
    print(f'Running {" ".join(cmd)}')
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    return float(p.stdout.strip())


@dataclass
class LoudnessInfo:
    integrated_lufs: float
    true_peak_dbtp: float
    range_lu: float
    threshold_lufs: float


def get_loudness_info(input_path: str) -> LoudnessInfo:
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",
        "-filter:a",
        "loudnorm=print_format=json",
        "-y",
        "-f",
        "null",
        "/dev/null",
    ]
    print(f"Running {' '.join(cmd)}")
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        check=True,
    )

    output = p.stdout
    reference_index = output.find("input_thresh")
    if reference_index == -1:
        print("Could not find input_thresh in loudnorm output. Stdout:")
        print(output)
        raise ValueError("Could not find loudnorm output.")

    start_bracket_index = output.rfind("{", None, reference_index)
    end_bracket_index = output.find("}", reference_index)
    loudnorm_data_raw = output[start_bracket_index : end_bracket_index + 1]

    loudnorm_data = json.loads(loudnorm_data_raw)
    input_i = loudnorm_data["input_i"]
    input_tp = loudnorm_data["input_tp"]
    input_lra = loudnorm_data["input_lra"]
    input_thresh = loudnorm_data["input_thresh"]

    return LoudnessInfo(
        integrated_lufs=float(input_i),
        true_peak_dbtp=float(input_tp),
        range_lu=float(input_lra),
        threshold_lufs=float(input_thresh),
    )


def get_audio_sampling_rate(input_path: str) -> int:
    # ffprobe -hide_banner -loglevel panic -show_streams -of json '.\P2 S11 N2 Another Day of Sun.mxf'
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "panic",
        "-show_streams",
        "-of",
        "json",
        input_path,
    ]
    print(f'Running {" ".join(cmd)}')
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    data = json.loads(p.stdout)
    found_sampling_rates: Set[int] = set()
    for stream in data["streams"]:
        if stream["codec_type"] == "audio":
            found_sampling_rates.add(int(stream["sample_rate"]))
    if len(found_sampling_rates) != 1:
        raise ValueError(f"Found more than one sampling rate: {found_sampling_rates}")
    return found_sampling_rates.pop()


def run_ffmpeg(
    input_path: str,
    height: Optional[int],
    width: Optional[int],
    crf: int,
    audio_bitrate_kb: int,
    audio_norm_db: int,
    audio_limiter: bool,
    target_size_mb: Optional[int],
) -> None:
    conforming_args = ["-pix_fmt", "yuv420p"]
    x264_args = ["-c:v", "libx264"]
    audio_args = ["-c:a", "aac", "-b:a", f"{audio_bitrate_kb}k"]
    suffixes = [f"a{audio_bitrate_kb}"]

    if height or width:
        suf = ""
        if height:
            suf += f"h{height}"
        if width:
            suf += f"w{width}"

        x264_args.extend(["-filter:v", f"scale={width or -2}:{height or -2}"])
        suffixes.append(suf)

    if target_size_mb:
        suffixes.append(f"tgt{target_size_mb}m")

        duration_secs = get_input_duration_secs(input_path)
        print(f"Input duration: {duration_secs} seconds")

        target_size_kbits = target_size_mb * 1024 * 8
        target_bitrate_kbits = int(
            target_size_kbits / duration_secs - audio_bitrate_kb - 32
        )
        x264_args.extend([f"-b:v", f"{target_bitrate_kbits}k"])

        first_pass_args = conforming_args + x264_args
        first_pass_args.extend(["-pass", "1"])
        first_pass_args.extend(["-f", "null", "-y", "/dev/null"])

        cmd = ["ffmpeg", "-i", input_path] + first_pass_args
        print(f'Running {" ".join(cmd)}')
        subprocess.run(cmd, check=True)

        x264_args.extend(["-pass", "2"])

    else:
        suffixes.append(f"crf{crf}")
        x264_args.extend(["-crf", str(crf)])

    if audio_norm_db:
        if audio_norm_db > 0:
            audio_norm_db = -audio_norm_db

        loudness_info = get_loudness_info(input_path)

        volume_delta = audio_norm_db - loudness_info.integrated_lufs
        headroom = -loudness_info.true_peak_dbtp - 0.5
        if volume_delta > headroom and not audio_limiter:
            print(
                f"Volume delta {volume_delta} exceeds headroom {headroom} and limiter disabled"
            )
            volume_delta = headroom
        new_lufs = loudness_info.integrated_lufs + volume_delta
        print(f"Will shift volume from {loudness_info.integrated_lufs} LUFS to {new_lufs} LUFS")

        if not audio_limiter:
            audio_args.extend(["-filter:a", f"volume={volume_delta}dB"])
        else:
            original_sample_rate = get_audio_sampling_rate(input_path)
            resampling_sample_rate = 4 * original_sample_rate
            filter_chain = [
                f"aresample={resampling_sample_rate}",
                f"volume={volume_delta}dB",
                "alimiter=limit=-0.5dB:level=0:attack=5:release=100",
                f"aresample={original_sample_rate}",
            ]
            audio_args.extend(["-filter:a", ",".join(filter_chain)])

        suffixes.append(f"{-round(new_lufs)}lufs")
        if audio_limiter:
            suffixes.append("lim")

    output_base = os.path.splitext(input_path)[0]
    output_suffixes_str = "".join("." + x for x in suffixes)
    output_path = f"{output_base}{output_suffixes_str}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        *conforming_args,
        *x264_args,
        *audio_args,
        output_path,
    ]
    print(f'Running {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def main():
    parser = ArgumentParser(description="Runs ffmpeg for me.")
    parser.add_argument("--crf", type=int, default=17, help="CRF value for video")
    parser.add_argument(
        "--audio-bitrate-kb", type=int, default=160, help="Audio bitrate in kbps"
    )
    parser.add_argument(
        "--audio-norm-db", type=int, default=16, help="Audio normalization in dB"
    )
    parser.add_argument(
        "--audio-limiter", action="store_true", help="Enable audio limiter"
    )
    parser.add_argument("--height", type=int, help="target height in pixels")
    parser.add_argument("--width", type=int, help="target width in pixels")
    parser.add_argument("--target-size-mb", type=int, help="Target size in MB")
    parser.add_argument("input", nargs="+", help="Input file(s)")
    args = parser.parse_args()

    raw_input_paths: List[str] = args.input
    input_paths: List[str] = []
    # Fail fast if any input file does not exist
    for input_path in raw_input_paths:
        if "*" in input_path:
            input_paths_chunk = glob.glob(input_path)
            input_paths.extend(input_paths_chunk)
        elif not os.path.exists(input_path):
            print(f"Input file {input_path} does not exist")
            sys.exit(1)
        else:
            input_paths.append(input_path)

    for input_path in input_paths:
        run_ffmpeg(
            input_path,
            args.height,
            args.width,
            args.crf,
            args.audio_bitrate_kb,
            args.audio_norm_db,
            args.audio_limiter,
            args.target_size_mb,
        )


if __name__ == "__main__":
    main()
