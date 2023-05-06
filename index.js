import React, { useEffect } from 'react'

const BUFFER_SIZE = 4194304
const WORKGROUP_SIZE = 128

let measurements = []

const init = async () => {
  // Check if the browser supports web gpu.
  if (!navigator.gpu) {
    throw Error(`WebGPU not supported.`)
  }

  // Request an adapter. An adapter represents a physical device (e.g. GPU)
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    throw Error(`Couldn't request WebGPU adapter.`)
  }

  // Request a device. A device is an abstraction in which a
  // single application can access gpu resources in a compartamentalized way.
  const device = await adapter.requestDevice()
  
  const shader = `
    @group(0) @binding(0)
    var<storage> seqs: array<u32>;

    @group(0) @binding(1)
    var<storage> seqs2: array<u32>;

    @group(0) @binding(2)
    var<storage> vals: array<u32>;

    @group(0) @binding(3)
    var<storage> vals2: array<u32>;

    @group(0) @binding(4)
    var<storage, read_write> outputSeqs: array<u32>;

    @group(0) @binding(5)
    var<storage, read_write> outputVals: array<u32>;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn main(
      @builtin(global_invocation_id)
      global_id : vec3u,
    ) {
      if (global_id.x >= ${BUFFER_SIZE}) {
        return;
      }

      let seq = seqs[global_id.x];
      let seq2 = seqs2[global_id.x];
      let val = vals[global_id.x];
      let val2 = vals2[global_id.x];

      if (seq2 > seq || (seq2 == seq && val2 > val)) {
        outputSeqs[global_id.x] = seq2;
        outputVals[global_id.x] = val2;
      } else {
        outputSeqs[global_id.x] = seq;
        outputVals[global_id.x] = val;
      }
    }
  `

  const shaderModule = device.createShaderModule({
    label: 'CRDT Shader',
    code: shader,
  })

  const outputSeqs = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const stagingSeqsBuffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const outputVals = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const stagingValsBuffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const seqs = new Uint32Array(BUFFER_SIZE / 4)

  const seqsBuffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  const getRandomInt = (min, max) =>
    Math.floor(Math.random() * (max - min + 1)) + min

  for (let i = 0; i < BUFFER_SIZE; ++i) {
    seqs[i] = getRandomInt(0, 1000)
  }

  const seqs2 = new Uint32Array(BUFFER_SIZE / 4)

  const seqs2Buffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  for (let i = 0; i < BUFFER_SIZE; ++i) {
    seqs2[i] = getRandomInt(0, 1000)
  }

  const vals = new Uint32Array(BUFFER_SIZE / 4)

  const valsBuffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  for (let i = 0; i < BUFFER_SIZE; ++i) {
    vals[i] = getRandomInt(0, 1000)
  }

  const vals2 = new Uint32Array(BUFFER_SIZE / 4)

  const vals2Buffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  device.queue.writeBuffer(seqsBuffer, 0, seqs)
  device.queue.writeBuffer(seqs2Buffer, 0, seqs2)
  device.queue.writeBuffer(valsBuffer, 0, vals)
  device.queue.writeBuffer(vals2Buffer, 0, vals2)

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'CRDT bind group layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
        },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
        },
      },
    ],
  })

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  })

  const bindGroup = device.createBindGroup({
    label: 'CRDT Bind Group',
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: seqsBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: seqs2Buffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: valsBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: vals2Buffer,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: outputSeqs,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: outputVals,
        },
      },
    ],
  })

  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })

  const startTime = performance.now()

  const commandEncoder = device.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()

  passEncoder.setPipeline(computePipeline)
  passEncoder.setBindGroup(0, bindGroup)
  const workgroupCount = Math.ceil(BUFFER_SIZE / WORKGROUP_SIZE)
  passEncoder.dispatchWorkgroups(workgroupCount)
  passEncoder.end()

  // Copy output buffer to staging buffer
  commandEncoder.copyBufferToBuffer(
    outputSeqs,
    0, // Source offset
    stagingSeqsBuffer,
    0, // Destination offset
    BUFFER_SIZE
  )

  commandEncoder.copyBufferToBuffer(
    outputVals,
    0, // Source offset
    stagingValsBuffer,
    0, // Destination offset
    BUFFER_SIZE
  )

  // End frame by passing array of command buffers to command queue for execution
  device.queue.submit([commandEncoder.finish()])

  // map staging buffer to read results back to JS
  await stagingSeqsBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    BUFFER_SIZE // Length
  )

  await stagingValsBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    BUFFER_SIZE // Length
  )

  const copySeqsArrayBuffer = stagingSeqsBuffer.getMappedRange(0, BUFFER_SIZE)
  const seqsData = copySeqsArrayBuffer.slice()
  stagingSeqsBuffer.unmap()

  const copyValsArrayBuffer = stagingValsBuffer.getMappedRange(0, BUFFER_SIZE)
  const valsData = copyValsArrayBuffer.slice()
  stagingValsBuffer.unmap()

  const endTime = performance.now()
  const elapsedTime = endTime - startTime
  // console.log(`Function took ${elapsedTime} milliseconds to execute.`)
  measurements.push(elapsedTime)

  // console.log(new Uint32Array(seqsData))
  // console.log(new Uint32Array(valsData))

  // _END_
}

const initLoop = async () => {
  for (let i = 0 ; i < 100; ++i) {
    for (let i = 0; i < 100; ++i) {
      await init()
    }

    // reduce the measurements down to an average
    // console log how many milliseconds it took to execute
    // log how many iterations happened to.
    const sum = measurements.reduce((a, b) => a + b, 0)
    const avg = sum / measurements.length
    console.log(`Function was ran ${measurements.length} times.`)
    console.log(`Function took ${avg} milliseconds to execute.`)
    console.log(`${BUFFER_SIZE / 4} operations merged per function.`)

    measurements = []
  }
}

let cpuMeasurements = []

const initCpu = async () => {
  const arrSize = BUFFER_SIZE / 4

  const getRandomInt = (min, max) =>
    Math.floor(Math.random() * (max - min + 1)) + min

  const seqs = new Uint32Array(arrSize)
  const seqs2 = new Uint32Array(arrSize)
  const vals = new Uint32Array(arrSize)
  const vals2 = new Uint32Array(arrSize)
  const outputSeqs = new Uint32Array(arrSize)
  const outputVals = new Uint32Array(arrSize)

  for (let i = 0; i < arrSize; ++i) {
    seqs[i] = getRandomInt(0, 1000)
    seqs2[i] = getRandomInt(0, 1000)
    vals[i] = getRandomInt(0, 1000)
    vals2[i] = getRandomInt(0, 1000)
  }

  const startTime = performance.now()

  for (let i = 0; i < arrSize; ++i) {
    if (seqs2[i] > seqs[i] || (seqs2[i] === seqs[i] && vals2[i] > vals[i])) {
      outputSeqs[i] = seqs2[i]
      outputVals[i] = vals2[i]
    } else {
      outputSeqs[i] = seqs[i]
      outputVals[i] = vals[i]
    }
  }

  const endTime = performance.now()
  const elapsedTime = endTime - startTime
  // console.log(`Function took ${elapsedTime} milliseconds to execute.`)
  cpuMeasurements.push(elapsedTime)
}

const initCpuLoop = async () => {
  for (let i = 0 ; i < 100; ++i) {
    for (let i = 0; i < 100; ++i) {
      await initCpu()
    }

    // reduce the measurements down to an average
    // console log how many milliseconds it took to execute
    // log how many iterations happened to.
    const sum = cpuMeasurements.reduce((a, b) => a + b, 0)
    const avg = sum / cpuMeasurements.length
    console.log(`Function was ran ${cpuMeasurements.length} times.`)
    console.log(`Function took ${avg} milliseconds to execute.`)
    console.log(`${BUFFER_SIZE / 4} operations merged per function.`)

    cpuMeasurements = []
  }
}

const WebGPU2 = () => {
  useEffect(() => {
    initLoop()
    // initCpuLoop()
  }, [])

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
       }}
    >
      <h1>WebGPU CRDT</h1>
    </div>
  )
}

export default WebGPU2
