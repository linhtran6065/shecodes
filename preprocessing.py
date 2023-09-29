from datasets import load_dataset, Audio


# rename, đưa hết vào flder data, tạo file metadata
# đưa hết audio vào folder data
ds = load_dataset("audio")

ds = ds.cast_column("audio", Audio(sampling_rate=16000))

print(ds)

# ds.push_to_hub("linhtran92/asr_data", token="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")

#, data_dir="D:\Desktop\shecodes\data"