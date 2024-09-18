# Introduction to Quantum Exact Simulation with Intel® FPGA

![](https://imageio.forbes.com/specials-images/imageserve/65664c6a3a659c7a14184544/Quantum-Computer/960x0.jpg?height=1225&width=711&fit=bounds){ align=right width=200 }

Quantum computing exact simulations involve the use of classical computers to model and analyze the behavior of quantum systems and quantum algorithms, bridging the gap between theoretical quantum mechanics and practical quantum computing technologies. These simulations are crucial for developing, testing, and optimizing quantum algorithms before they are run on actual quantum hardware, which is often limited in availability and capability.

## Existing simulators

There are several [quantum simulators](https://quantiki.org/wiki/list-qc-simulators) available that vary in their approach, capabilities, and the scale of systems they can simulate. Among the most famous one, you have:

- **Qiskit Aer**: Developed by IBM, Qiskit Aer is an open-source simulator that allows users to perform realistic simulations of quantum circuits, complete with noise models and resource estimations. It helps in understanding how quantum algorithms will perform on real quantum hardware.

- **Cirq**: Google's Cirq is another open-source framework designed to simulate and test quantum algorithms on local machines. It is particularly tailored for noisy intermediate-scale quantum (NISQ) computers.

- **cuQuantum**: NVIDIA's specialized SDK (Software Development Kit) for accelerating quantum computing simulations on GPUs. Announced and developed by NVIDIA, this toolkit is designed to harness the parallel processing capabilities of GPUs to speed up the simulation of quantum circuits and quantum systems. cuQuantum targets both researchers and developers in the field of quantum computing, providing tools and libraries optimized for NVIDIA's GPU architecture.

Simulations play a dual role in the quantum computing landscape. They are instrumental in:

- **Algorithm Development**: By simulating the outcomes of quantum algorithms, researchers can identify potential improvements and optimizations without needing access to quantum processors.

- **Hardware Design**: Simulations help predict how quantum devices might perform in the real world, aiding in the design and construction of more effective quantum hardware.

## Hardware accelerators (HA)

Hardware accelerators provide significant advantages for simulating quantum systems due to their powerful parallel processing capabilities. In the field of quantum computing, where simulating quantum phenomena on classical computers can be computationally intensive, HA help in addressing some of these challenges by accelerating calculations.

- **Parallelism**: Quantum simulations involve operations on large vectors and matrices since the state of a quantum system is represented by a state vector in a complex vector space, and operations on these states are represented by matrices. For example, GPUs are well-suited for these tasks due to their highly parallel architecture, allowing for faster processing of these large-scale linear algebra operations compared to traditional CPUs.

- **Scalability**: While the exponential growth of the quantum state space with each added qubit remains a challenge, HA help push the boundaries of what size systems can be simulated. They enable researchers to simulate slightly larger quantum systems than would be feasible with CPUs alone.

- **Efficiency**: For specific types of quantum simulations, such as those involving tensor networks or state vector simulations, HA can significantly speed up the computation. This efficiency is crucial for exploring more complex quantum algorithms and systems within practical time frames.

- Although less known than GPUs, **FPGAs (Field-Programmable Gate Arrays)** represent a growing area of interest in the field of quantum computing due to FPGAs' unique properties.

- FPGAs are integrated circuits that can be configured by the user after manufacturing, allowing for highly specialized hardware setups tailored to specific computational tasks.

![fpga](https://www.bittware.com/files/520N-MX-800px.svg)

### Why using FPGAs for Quantum Simulations?

- **Customizability and Reconfigurability**: Unlike CPUs and GPUs, which have fixed architectures, FPGAs can be programmed to create custom hardware configurations. This allows for the optimization of specific algorithms or processes, which can be particularly beneficial for quantum simulations, where different algorithms might benefit from different hardware optimizations.

- **Parallel Processing**: FPGAs can be designed to handle parallel computations natively using pipeline parallelism.

- **Low Latency and High Throughput**: FPGAs can provide lower latency than CPUs and GPUs because they can be programmed to execute tasks without the overhead of an operating system or other software layers. This makes them ideal for real-time processing and simulations.

- **Energy Efficiency**: FPGAs can be more energy-efficient than GPUs and CPUs for certain tasks because they can be stripped down to only the necessary components required for a specific computation, reducing power consumption.

## [Intel® FPGA SDK & oneAPI for FPGA](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.3c0top) 

- The [Intel® FPGA Software Development Kit (SDK)](https://www.intel.com/content/www/us/en/docs/programmable/683846/22-4/overview.html) provides a comprehensive set of development tools and libraries specifically designed to facilitate the design, creation, testing, and deployment of applications on Intel's FPGA hardware. The SDK includes tools for both high-level and low-level programming, including support for hardware description languages like VHDL and Verilog, as well as higher-level abstractions using OpenCL or HLS (High-Level Synthesis). This makes it easier for developers to leverage the power of FPGAs without needing deep expertise in hardware design.

![](./images/Intel-oneAPI-logo-686x600.jpg){ align=right width=200 }

- [Intel® oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.3c0top) is a unified programming model designed to simplify development across diverse computing architectures—CPUs, GPUs, FPGAs, and other accelerators. The oneAPI for FPGA component specifically targets the optimization and utilization of Intel FPGAs. It allows developers to use a single, consistent programming model to target various hardware platforms, facilitating easier code reuse and system integration. oneAPI includes specialized libraries and tools that enable developers to maximize the performance of their applications on Intel FPGAs while maintaining a high level of productivity and portability.

In this course, you will learn to:

- How to use Meluxina's FPGA, i.e., Intel® FPGA

- How to exploit FPGA for quantum simulation

- How to take advantage of the Intel® oneAPI to code quantum circuit

!!! danger "Remark"
    This course is not intended to be exhaustive. In addition, the described tools and features are constantly evolving. We try our best to keep it up to date. 

## Who is the course for ?

- This course is for students, researchers, enginners wishing to discover how to use oneAPI to program FPGA in this fantastic fields which is **Quantum Computing**. Participants should still have some experience with Python & modern C++ (e.g., [Lambdas](https://en.cppreference.com/w/cpp/language/lambda), [class deduction templates](https://en.cppreference.com/w/cpp/language/class_template_argument_deduction)).

- This course is **NOT** a Quantum Computing course but intends to show you how to use QC simulation on Meluxina's FPGA.

- We strongly recommend to interested particpants this [CERN online course](https://indico.cern.ch/event/970903/).

## About this course

This course has been developed by the **Supercomputing Application Services** group at LuxProvide. 



