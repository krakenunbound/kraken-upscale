//! Native RRDBNet `.pth` importer.
//!
//! PyTorch state dictionaries are read directly and assembled into an
//! in-memory ONNX Runtime graph without temporary files or preparation steps.

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

use candle_core::{DType, Device, Tensor as CandleTensor};
use ort::editor::{Graph, Model, Node, Opset, ONNX_DOMAIN};
use ort::ep::{ExecutionProviderDispatch, CPU, CUDA};
use ort::memory::Allocator;
use ort::operator::Attribute;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Outlet, Shape, SymbolicDimensions, Tensor, TensorElementType, ValueType};

type State = HashMap<String, CandleTensor>;

pub fn load_rrdb(path: &Path) -> Result<(Session, u32, String), String> {
    let tensors = read_checkpoint(path)?;
    if tensors.is_empty() {
        return Err("PyTorch file contained no tensors".into());
    }

    let mut state: State = tensors.into_iter().collect();
    normalize_prefix(&mut state);
    let layout = Layout::detect(&state)?;
    let scale = layout.scale(&state)?;
    let blocks = layout.block_count(&state)?;

    let mut graph = Graph::new().map_err(err("creating graph"))?;
    graph
        .set_inputs([Outlet::new(
            "input",
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape: Shape::new([-1, 3, -1, -1]),
                dimension_symbols: SymbolicDimensions::empty(4),
            },
        )])
        .map_err(err("setting graph input"))?;
    graph
        .set_outputs([Outlet::new(
            "output",
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape: Shape::new([-1, 3, -1, -1]),
                dimension_symbols: SymbolicDimensions::empty(4),
            },
        )])
        .map_err(err("setting graph output"))?;

    // Initializers retain their state-dict names, which keeps graph diagnostics
    // readable and avoids a second full copy of the key set.
    for (name, tensor) in &state {
        let tensor = tensor
            .to_dtype(DType::F32)
            .map_err(|e| format!("converting {name} to f32: {e}"))?;
        let shape: Vec<i64> = tensor.dims().iter().map(|&d| d as i64).collect();
        let data = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("reading tensor {name}: {e}"))?;
        let value = owned_tensor(Shape::new(shape), data)
            .map_err(|e| format!("creating initializer {name}: {e}"))?;
        graph
            .add_initializer(name, value, false)
            .map_err(|e| format!("adding initializer {name}: {e}"))?;
    }

    // Constants shared by every residual block and upsampling stage.
    graph
        .add_initializer(
            "kraken.residual_scale",
            owned_tensor(Shape::new([]), vec![0.2f32]).map_err(err("creating residual scale"))?,
            false,
        )
        .map_err(err("adding residual scale"))?;
    graph
        .add_initializer(
            "kraken.resize_scales",
            owned_tensor(Shape::new([4]), vec![1.0f32, 1.0, 2.0, 2.0])
                .map_err(err("creating resize scales"))?,
            false,
        )
        .map_err(err("adding resize scales"))?;

    let mut b = Builder {
        graph: &mut graph,
        state: &state,
        serial: 0,
    };
    let first = layout.first();
    let network_input = match layout.input_block(scale) {
        1 => "input".to_string(),
        block => b.space_to_depth("input", block)?,
    };
    let feature = b.conv(&network_input, first)?;
    let mut trunk = feature.clone();
    for block in 0..blocks {
        trunk = b.rrdb(&trunk, &layout.block(block))?;
    }
    trunk = b.conv(&trunk, layout.trunk(blocks))?;
    let mut feature = b.add(&feature, &trunk)?;

    for up in layout.upsample_layers(scale) {
        feature = b.resize2x(&feature)?;
        feature = b.conv(&feature, up)?;
        feature = b.lrelu(&feature)?;
    }
    feature = b.conv(&feature, layout.hr(scale))?;
    feature = b.lrelu(&feature)?;
    let output = b.conv_named(&feature, layout.last(scale), "output")?;
    debug_assert_eq!(output, "output");

    let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22).map_err(err("creating opset"))?])
        .map_err(err("creating model"))?;
    model.add_graph(graph).map_err(err("adding graph"))?;

    // Mirror the .onnx provider policy: demand CUDA loudly and fall back to CPU
    // only if registration genuinely fails. `CUDA::is_available()` reports that
    // ORT was *compiled* with CUDA support, not that this session got it, so
    // labelling from it could claim CUDA while inference silently ran on CPU.
    let builder = |providers: Vec<ExecutionProviderDispatch>| {
        Session::builder()
            .map_err(err("session builder"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("optimization level: {e}"))?
            .with_execution_providers(providers)
            .map_err(|e| format!("execution providers: {e}"))
    };

    // Provider is decided before the model is consumed, so a CUDA failure can
    // still fall back without having to rebuild the graph.
    let (options, provider) = match builder(vec![CUDA::default().build().error_on_failure()]) {
        Ok(options) => (options, "CUDA"),
        Err(cuda_err) => {
            eprintln!("CUDA unavailable for checkpoint import, using CPU: {cuda_err}");
            (builder(vec![CPU::default().build()])?, "CPU")
        }
    };

    let session = model
        .into_session(&options)
        .map_err(err("creating RRDB session"))?;
    Ok((session, scale, provider.into()))
}

fn err(context: &'static str) -> impl FnOnce(ort::Error) -> String {
    move |e| format!("{context}: {e}")
}

fn owned_tensor(shape: Shape, data: Vec<f32>) -> Result<Tensor<f32>, ort::Error> {
    let mut tensor = Tensor::<f32>::new(&Allocator::default(), shape)?;
    let (_, values) = tensor.extract_tensor_mut();
    values.copy_from_slice(&data);
    Ok(tensor)
}

fn read_checkpoint(path: &Path) -> Result<Vec<(String, CandleTensor)>, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("opening PyTorch checkpoint {}: {e}", path.display()))?;
    let mut signature = [0u8; 4];
    file.read_exact(&mut signature)
        .map_err(|e| format!("reading PyTorch checkpoint signature: {e}"))?;
    if signature == *b"PK\x03\x04" {
        candle_core::pickle::read_all(path).map_err(|e| {
            format!(
                "reading zipped PyTorch state dictionary {}: {e}",
                path.display()
            )
        })
    } else {
        read_legacy_checkpoint(path)
    }
}

/// Reads PyTorch's pre-1.6 stream format. Community ESRGAN checkpoints still
/// commonly use this exact format: pickle metadata followed by keyed raw
/// storage blocks. The downloaded file is read directly and never rewritten.
fn read_legacy_checkpoint(path: &Path) -> Result<Vec<(String, CandleTensor)>, String> {
    use candle_core::pickle::{Object, Stack, TensorInfo};

    let file = std::fs::File::open(path)
        .map_err(|e| format!("opening legacy checkpoint {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);

    // PROTO 2 + LONG1(10) + PyTorch's fixed serialization magic + STOP.
    // Candle's generic LONG1 reader is intentionally i64-sized, while this
    // magic is 80-bit, so validate and consume it without interpreting it.
    const LEGACY_MAGIC: [u8; 15] = [
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, 0x2e,
    ];
    let mut magic = [0u8; LEGACY_MAGIC.len()];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("reading legacy PyTorch header: {e}"))?;
    if magic != LEGACY_MAGIC {
        return Err("unsupported legacy PyTorch serialization header".into());
    }

    fn object(reader: &mut impl std::io::BufRead) -> Result<Object, String> {
        let mut stack = Stack::empty();
        stack
            .read_loop(reader)
            .map_err(|e| format!("reading legacy pickle: {e}"))?;
        stack
            .finalize()
            .map_err(|e| format!("finishing legacy pickle: {e}"))
    }

    let protocol = object(&mut reader)?;
    if !matches!(protocol, Object::Int(1001)) {
        return Err(format!("unsupported legacy PyTorch protocol {protocol:?}"));
    }
    let _system_info = object(&mut reader)?;
    let state = object(&mut reader)?;
    let storage_keys = object(&mut reader)?;

    let entries = match state {
        Object::Dict(entries) => entries,
        other => {
            return Err(format!(
                "legacy checkpoint state is not a dictionary: {other:?}"
            ))
        }
    };
    let mut infos = Vec::<TensorInfo>::new();
    for (name, value) in entries {
        match value.into_tensor_info(name, Path::new("")) {
            Ok(Some(info)) => infos.push(info),
            Ok(None) => {}
            Err(e) => return Err(format!("reading legacy tensor metadata: {e}")),
        }
    }
    if infos.is_empty() {
        return Err("legacy checkpoint contained no tensor metadata".into());
    }

    let storage_keys = match storage_keys {
        Object::List(keys) => keys
            .into_iter()
            .map(|key| match key {
                Object::Unicode(key) => Ok(key),
                other => Err(format!("invalid legacy storage key {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        other => return Err(format!("legacy storage index is not a list: {other:?}")),
    };

    let mut storage_meta = HashMap::new();
    for info in &infos {
        storage_meta
            .entry(info.path.trim_start_matches('/').to_string())
            .or_insert((info.dtype, info.storage_size));
    }
    let mut storages = HashMap::<String, Vec<u8>>::new();
    for key in storage_keys {
        let (dtype, expected) = storage_meta
            .get(&key)
            .copied()
            .ok_or_else(|| format!("storage {key} has no tensor metadata"))?;
        let mut count = [0u8; 8];
        reader
            .read_exact(&mut count)
            .map_err(|e| format!("reading storage {key} length: {e}"))?;
        let count = u64::from_le_bytes(count) as usize;
        if count < expected {
            return Err(format!(
                "storage {key} is shorter than declared ({count} < {expected})"
            ));
        }
        let byte_len = count
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| format!("storage {key} is too large"))?;
        let mut bytes = vec![0u8; byte_len];
        reader
            .read_exact(&mut bytes)
            .map_err(|e| format!("reading storage {key}: {e}"))?;
        storages.insert(key, bytes);
    }

    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos {
        let key = info.path.trim_start_matches('/');
        let bytes = storages
            .get(key)
            .ok_or_else(|| format!("tensor {} references missing storage {key}", info.name))?;
        let start = info
            .layout
            .start_offset()
            .checked_mul(info.dtype.size_in_bytes())
            .ok_or_else(|| format!("tensor {} has an invalid storage offset", info.name))?;
        let slice = bytes
            .get(start..)
            .ok_or_else(|| format!("tensor {} starts outside storage {key}", info.name))?;
        let fortran = info.layout.is_fortran_contiguous();
        if !info.layout.is_contiguous() && !fortran {
            return Err(format!("tensor {} is non-contiguous", info.name));
        }
        let rank = info.layout.shape().rank();
        let elem_count = info.layout.shape().elem_count();
        let mut tensor = match info.dtype {
            DType::F32 => {
                let byte_len = elem_count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| format!("tensor {} is too large", info.name))?;
                let raw = slice
                    .get(..byte_len)
                    .ok_or_else(|| format!("tensor {} extends beyond storage {key}", info.name))?;
                let values = raw
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                CandleTensor::from_vec(values, info.layout.shape().clone(), &Device::Cpu)
                    .map_err(|e| format!("reading tensor {}: {e}", info.name))?
            }
            dtype => {
                return Err(format!(
                    "tensor {} uses unsupported legacy storage type {dtype:?}",
                    info.name
                ));
            }
        };
        if rank > 1 && fortran {
            let reversed: Vec<_> = info.layout.dims().iter().rev().copied().collect();
            tensor = tensor
                .reshape(reversed)
                .and_then(|t| t.permute((0..rank).rev().collect::<Vec<_>>()))
                .map_err(|e| format!("restoring tensor {} layout: {e}", info.name))?;
        }
        tensors.push((info.name, tensor));
    }
    Ok(tensors)
}

fn normalize_prefix(state: &mut State) {
    for prefix in ["params_ema.", "params.", "state_dict."] {
        if !state.is_empty() && state.keys().all(|k| k.starts_with(prefix)) {
            let old = std::mem::take(state);
            *state = old
                .into_iter()
                .map(|(k, v)| (k.trim_start_matches(prefix).to_string(), v))
                .collect();
        }
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Old,
    New,
}

impl Layout {
    fn detect(state: &State) -> Result<Self, String> {
        if state.contains_key("model.0.weight") {
            Ok(Self::Old)
        } else if state.contains_key("conv_first.weight") {
            Ok(Self::New)
        } else {
            Err("unsupported .pth topology: expected an old ESRGAN or RealESRGAN RRDBNet state dictionary".into())
        }
    }

    fn first(self) -> &'static str {
        match self {
            Self::Old => "model.0",
            Self::New => "conv_first",
        }
    }

    fn block(self, index: usize) -> String {
        match self {
            Self::Old => format!("model.1.sub.{index}"),
            Self::New => format!("body.{index}"),
        }
    }

    fn block_count(self, state: &State) -> Result<usize, String> {
        let mut count = 0;
        loop {
            let key = match self {
                Self::Old => format!("model.1.sub.{count}.RDB1.conv1.0.weight"),
                Self::New => format!("body.{count}.rdb1.conv1.weight"),
            };
            if !state.contains_key(&key) {
                break;
            }
            count += 1;
        }
        (count > 0)
            .then_some(count)
            .ok_or_else(|| "RRDBNet state dictionary contains no residual blocks".into())
    }

    fn trunk(self, blocks: usize) -> String {
        match self {
            Self::Old => format!("model.1.sub.{blocks}"),
            Self::New => "conv_body".into(),
        }
    }

    fn scale(self, state: &State) -> Result<u32, String> {
        match self {
            Self::New => {
                let input_channels = state
                    .get("conv_first.weight")
                    .and_then(|tensor| tensor.dims().get(1).copied())
                    .ok_or_else(|| {
                        "conv_first.weight has no input-channel dimension".to_string()
                    })?;
                match input_channels {
                    3 => Ok(4),
                    12 => Ok(2),
                    48 => Ok(1),
                    channels => Err(format!(
                        "unsupported RealESRGAN input channel count {channels}"
                    )),
                }
            }
            Self::Old => {
                if state.contains_key("model.13.weight") {
                    Ok(8)
                } else if state.contains_key("model.10.weight") {
                    Ok(4)
                } else if state.contains_key("model.7.weight") {
                    Ok(2)
                } else {
                    Err("unsupported old-ESRGAN upsampling layout".into())
                }
            }
        }
    }

    fn upsample_layers(self, scale: u32) -> Vec<String> {
        match (self, scale) {
            (Self::Old, 2 | 4 | 8) => (0..scale.ilog2())
                .map(|index| format!("model.{}", 3 + index * 3))
                .collect(),
            (Self::New, 1 | 2 | 4) => vec!["conv_up1".into(), "conv_up2".into()],
            _ => Vec::new(),
        }
    }

    fn input_block(self, scale: u32) -> i64 {
        match (self, scale) {
            (Self::New, 1) => 4,
            (Self::New, 2) => 2,
            _ => 1,
        }
    }

    fn hr(self, scale: u32) -> String {
        match self {
            Self::Old => format!("model.{}", 3 * scale.ilog2() + 2),
            Self::New => "conv_hr".into(),
        }
    }

    fn last(self, scale: u32) -> String {
        match self {
            Self::Old => format!("model.{}", 3 * scale.ilog2() + 4),
            Self::New => "conv_last".into(),
        }
    }
}

struct Builder<'a> {
    graph: &'a mut Graph,
    state: &'a State,
    serial: usize,
}

impl Builder<'_> {
    fn name(&mut self, kind: &str) -> String {
        self.serial += 1;
        format!("kraken.{kind}.{}", self.serial)
    }

    fn add_node(
        &mut self,
        op: &str,
        inputs: &[String],
        output: String,
        attrs: Vec<Attribute>,
    ) -> Result<String, String> {
        let node_name = self.name(&op.to_ascii_lowercase());
        self.graph
            .add_node(
                Node::new(op, ONNX_DOMAIN, node_name, inputs, [output.as_str()], attrs)
                    .map_err(|e| format!("creating {op} node: {e}"))?,
            )
            .map_err(|e| format!("adding {op} node: {e}"))?;
        Ok(output)
    }

    fn conv(&mut self, input: &str, prefix: impl AsRef<str>) -> Result<String, String> {
        let output = self.name("conv_out");
        self.conv_named(input, prefix, &output)
    }

    fn conv_named(
        &mut self,
        input: &str,
        prefix: impl AsRef<str>,
        output: &str,
    ) -> Result<String, String> {
        let prefix = prefix.as_ref();
        let weight = format!("{prefix}.weight");
        if !self.state.contains_key(&weight) {
            return Err(format!("state dictionary is missing {weight}"));
        }
        let bias = format!("{prefix}.bias");
        let mut inputs = vec![input.to_string(), weight];
        if self.state.contains_key(&bias) {
            inputs.push(bias);
        }
        self.add_node(
            "Conv",
            &inputs,
            output.to_string(),
            vec![
                Attribute::new("dilations", vec![1i64, 1]).map_err(err("conv dilations"))?,
                Attribute::new("group", 1i64).map_err(err("conv group"))?,
                Attribute::new("kernel_shape", vec![3i64, 3]).map_err(err("conv kernel"))?,
                Attribute::new("pads", vec![1i64, 1, 1, 1]).map_err(err("conv pads"))?,
                Attribute::new("strides", vec![1i64, 1]).map_err(err("conv strides"))?,
            ],
        )
    }

    fn lrelu(&mut self, input: &str) -> Result<String, String> {
        let output = self.name("lrelu_out");
        self.add_node(
            "LeakyRelu",
            &[input.to_string()],
            output,
            vec![Attribute::new("alpha", 0.2f32).map_err(err("leaky relu alpha"))?],
        )
    }

    fn add(&mut self, a: &str, b: &str) -> Result<String, String> {
        let output = self.name("add_out");
        self.add_node("Add", &[a.to_string(), b.to_string()], output, Vec::new())
    }

    fn residual(&mut self, original: &str, delta: &str) -> Result<String, String> {
        let scaled = self.name("scaled_out");
        let scaled = self.add_node(
            "Mul",
            &[delta.to_string(), "kraken.residual_scale".into()],
            scaled,
            Vec::new(),
        )?;
        self.add(original, &scaled)
    }

    fn concat(&mut self, values: &[String]) -> Result<String, String> {
        let output = self.name("concat_out");
        self.add_node(
            "Concat",
            values,
            output,
            vec![Attribute::new("axis", 1i64).map_err(err("concat axis"))?],
        )
    }

    fn resize2x(&mut self, input: &str) -> Result<String, String> {
        let output = self.name("resize_out");
        self.add_node(
            "Resize",
            &[
                input.to_string(),
                String::new(),
                "kraken.resize_scales".into(),
            ],
            output,
            // ONNX defaults to nearest-neighbour sampling. At an exact 2x
            // scale its default coordinate rule gives the pixel repetition
            // used by RRDBNet's nearest-neighbour upsampling stage.
            vec![],
        )
    }

    fn space_to_depth(&mut self, input: &str, block: i64) -> Result<String, String> {
        let output = self.name("space_to_depth_out");
        self.add_node(
            "SpaceToDepth",
            &[input.to_string()],
            output,
            vec![Attribute::new("blocksize", block).map_err(err("space-to-depth block"))?],
        )
    }

    fn rdb(&mut self, input: &str, prefix: &str, old: bool) -> Result<String, String> {
        let original = input.to_string();
        let mut features = vec![original.clone()];
        for conv in 1..=5 {
            let joined = self.concat(&features)?;
            let conv_prefix = if old {
                format!("{prefix}.conv{conv}.0")
            } else {
                format!("{prefix}.conv{conv}")
            };
            let mut out = self.conv(&joined, conv_prefix)?;
            if conv < 5 {
                out = self.lrelu(&out)?;
                features.push(out);
            } else {
                return self.residual(&original, &out);
            }
        }
        unreachable!()
    }

    fn rrdb(&mut self, input: &str, prefix: &str) -> Result<String, String> {
        let original = input.to_string();
        let old = self
            .state
            .contains_key(&format!("{prefix}.RDB1.conv1.0.weight"));
        let mut out = original.clone();
        for rdb in 1..=3 {
            let rdb_prefix = if old {
                format!("{prefix}.RDB{rdb}")
            } else {
                format!("{prefix}.rdb{rdb}")
            };
            out = self.rdb(&out, &rdb_prefix, old)?;
        }
        self.residual(&original, &out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "set KRAKEN_TEST_PTH to a real RRDBNet checkpoint"]
    fn imports_and_runs_real_checkpoint() {
        let path = std::env::var_os("KRAKEN_TEST_PTH")
            .map(std::path::PathBuf::from)
            .expect("KRAKEN_TEST_PTH is required");
        let (mut session, scale, _) = load_rrdb(&path).expect("import checkpoint");
        let input = Tensor::<f32>::from_array((
            [1usize, 3, 8, 8],
            vec![0.5f32; 3 * 8 * 8].into_boxed_slice(),
        ))
        .expect("input tensor");
        let outputs = session
            .run(ort::inputs!["input" => input])
            .expect("run imported graph");
        let (shape, _) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("extract output");
        assert_eq!(&shape[..], &[1, 3, 8 * scale as i64, 8 * scale as i64]);
    }
}
