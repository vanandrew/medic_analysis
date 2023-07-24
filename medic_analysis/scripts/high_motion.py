from pathlib import Path
import re
import pandas as pd

ASD_ADHD_DATASET = Path("/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")


def main():
    data = {}
    # loop through subjects
    i = 0
    for subject in ASD_ADHD_DATASET.glob("sub-*"):
        # loop over sessions
        for session in subject.glob("ses-*"):
            print(f"In {subject.name} {session.name}")
            if "wTOPUP" in session.name:
                isTOPUP = True
                session_name = session.name.replace("wTOPUP", "")
            else:
                isTOPUP = False
                session_name = session.name
            # check if movement directory exists
            movement_dir = session / "movement"
            if not movement_dir.exists():
                print(f"{session.name} does not have movement directory")
                continue
            # loop over dat files
            for dat_file in movement_dir.glob("*.dat"):
                match = re.search(r"_b(\d+)_", dat_file.name)
                if match:
                    run = f"bold{match.group(1)}"
                else:
                    raise ValueError(f"Filename malformed check the outputs at: {dat_file.as_posix()}")
                with open(dat_file, "r") as f:
                    lines = f.readlines()
                    rms_motion = float(lines[-1].strip().split(" ")[-1])
                # for this bold run grab the tmask
                tmask = list((session / run).glob("*tmask*"))[0]
                # load the tmask and compute the number of 1.0s
                with open(tmask, "r") as f:
                    lines = f.readlines()
                    num_frames = len(lines)
                    good_frames = int(sum([float(x.strip()) for x in lines]))
                data[i] = [subject.name, session_name, run, rms_motion, good_frames, num_frames, isTOPUP]
                i += 1
    df = pd.DataFrame.from_dict(
        data, "index", columns=["subject", "session", "run", "rms_motion", "good_frames", "num_frames", "isTOPUP"]
    )
    df.sort_values(by=["subject", "session", "run"], inplace=True)
    print(df)
    df.to_csv("high_motion.csv", index=False)
