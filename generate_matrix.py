import json, itertools

window_sizes = ["512","1024","2048"] 

percent_overlaps = ["0","10","20","30","40","50","60","70","80","90"]

window_types = ["tukey", "boxcar", "hann", "hamming", "blackman", "blackmanharris"]

scalings = ["density","spectrum"]

modes = ["psd","magnitude"]

resample_method = ["none", "librosa","polyphase"]

target_sr = ["22050"]

matrix = [
{
    "window_size":     ws,
    "percent_overlap": po,
    "window_type":     wt,
    "scaling":         sc,
    "mode":            md,
    "resample_method": rm,
    "target_sr":       ts
}
for ws, po, wt, sc, md, rm, ts in itertools.product(
    window_sizes, percent_overlaps, window_types, scalings, modes, resample_method, target_sr
)
]

# print(json.dumps(matrix))
    
print(f"Total combinations: {len(matrix)}")


# Save the matrix to a JSON file
with open("matrix.json", "w") as f:
    json.dump(matrix, f, indent=4)