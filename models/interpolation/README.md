# RIFE interpolation model

Place `RIFE_fp32_timestep.onnx` in this directory. Kraken loads it directly
through its bundled ONNX Runtime CUDA provider; no conversion step is required.

- Source: <https://huggingface.co/walterlow/RIFE_fp32_timestep>
- Pinned revision: `ee09066f9822f8b28b8477a1b4cc30f19d607590`
- File: `RIFE_fp32_timestep.onnx`
- SHA-256: `4DA60C1F20D42DBA4F21503140940AA48A488585EAD977532BF22E95A0319327`
- License: MIT, inherited from Practical-RIFE and ECCV2022-RIFE

The graph accepts two planar RGB frames in `[0, 1]` and a scalar timestep,
then produces the motion-aware frame at that phase. Dynamic frame dimensions
are supported and padded internally by the graph.
