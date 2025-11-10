from pathlib import Path 
import os 

def cutlass_profile_win(path: Path, cmd: str):
    import subprocess
    ncu_sections = [
        "ComputeWorkloadAnalysis", "InstructionStats", "LaunchStats",
        "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Chart",
        "MemoryWorkloadAnalysis_Tables", "Occupancy", "SchedulerStats",
        "SourceCounters", "SpeedOfLight", "WarpStateStats"
    ]
    section_flags = sum([["--section", s] for s in ncu_sections], [])
    cmds = [
        "ncu", "-o",
        str(path), *section_flags, "-f",
        cmd
    ]
    print(" ".join(cmds))
    subprocess.check_call(cmds, env=os.environ, shell=True)


if __name__ == "__main__":
    p = Path("/home/yy/Projects/cumm/cumm/conv/profile")
    cmd = "python /home/yy/Projects/cumm/cumm/conv/main_real.py"

    cutlass_profile_win(p, cmd)