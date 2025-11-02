import re
import glob
import pandas as pd

results = []

for file in glob.glob("output/slurm-*.out"):
    with open(file) as f:
        text = f.read()
        
        job_match = re.search(
            r"Parameters: LR=([\d.]+), OPT=(\w+), BS=(\d+), LOSS=([\w_]+), DO=([\d.]+), RDO=([\d.]+), UNITS=(\d+), LAYERS=(\d+), BIDIR=(\w+)", 
            text
        )
        
        f1_match = re.search(r"F1 Score \(OFF\): ([\d.]+)", text)
        macro_f1_match = re.search(r"Macro F1 Score: ([\d.]+)", text)
        
        if job_match and f1_match and macro_f1_match:
            lr = float(job_match.group(1))
            opt = job_match.group(2)
            bs = int(job_match.group(3))
            loss = job_match.group(4)
            dropout = float(job_match.group(5))
            rdo = float(job_match.group(6))
            units = int(job_match.group(7))
            layers = int(job_match.group(8))
            bidir = job_match.group(9) == "True"
            f1 = float(f1_match.group(1))
            macro_f1 = float(macro_f1_match.group(1)) # Capture the new value
            
            results.append({
                "file": file,
                "learning_rate": lr,
                "optimizer": opt,
                "batch_size": bs,
                "loss": loss,
                "dropout": dropout,
                "recurrent_dropout": rdo,
                "units": units,
                "layers": layers,
                "bidirectional": bidir,
                "f1_score_off": f1,
                "f1_score_macro": macro_f1 # Add the new value to the dictionary
            })

df = pd.DataFrame(results)
df = df.sort_values(by="f1_score_macro", ascending=False) 

# Save to CSV
df.to_csv("summary.csv", index=False)

print("Successfully created summary.csv with the following data:")
print(df)
