# Bleeding Edge Overlay - Latest AI/ML packages and experimental tools
# Apply via: overlays = [ science-flake.overlays.bleedingEdge ];
# 
# This overlay provides cutting-edge packages that may not be stable.
# Use for research and experimentation, not production.

final: prev: {
  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON WITH BLEEDING EDGE ML STACK
  # ════════════════════════════════════════════════════════════════════════════
  python-ml-bleeding = prev.python311.withPackages (ps: with ps; [
    # Core
    numpy
    scipy
    pandas
    polars  # Fast DataFrame
    
    # PyTorch bleeding edge
    torch
    torchvision
    torchaudio
    
    # Transformers ecosystem
    transformers
    datasets
    tokenizers
    accelerate
    safetensors
    
    # Scientific
    scikit-learn
    xgboost
    lightgbm
    
    # Visualization
    matplotlib
    seaborn
    plotly
    
    # Notebooks
    jupyterlab
    ipywidgets
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # LLAMA.CPP WITH CUDA
  # ════════════════════════════════════════════════════════════════════════════
  llama-cpp-cuda = prev.llama-cpp.override {
    cudaSupport = true;
    # Uncomment for specific GPU:
    # cudaArch = "86";  # RTX 30xx
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PYTORCH WITH SPECIFIC CUDA VERSION
  # ════════════════════════════════════════════════════════════════════════════
  # Note: This is a placeholder - actual CUDA version selection happens
  # at the nixpkgs config level, not overlay level for PyTorch
  
  # ════════════════════════════════════════════════════════════════════════════
  # BIOINFORMATICS BLEEDING EDGE - CORE
  # ════════════════════════════════════════════════════════════════════════════
  python-bio-bleeding = prev.python311.withPackages (ps: with ps; [
    # Single-cell
    scanpy
    anndata
    
    # Genomics
    biopython
    pysam
    pybedtools
    
    # ML for biology
    torch
    scikit-learn
    
    # Data
    numpy
    scipy
    pandas
    h5py
    zarr
    
    # Viz
    matplotlib
    seaborn
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # BIOINFORMATICS BLEEDING EDGE - ML FOR BIOLOGY
  # ════════════════════════════════════════════════════════════════════════════
  python-bioml-bleeding = prev.python311.withPackages (ps: with ps; [
    # Core ML
    torch
    torchvision
    
    # Transformers for biology
    transformers
    datasets
    tokenizers
    accelerate
    
    # Biology-specific
    biopython
    
    # Data processing
    numpy
    scipy
    pandas
    polars
    h5py
    zarr
    
    # Visualization
    matplotlib
    seaborn
    plotly
    
    # Experiment tracking
    tensorboard
    wandb
    
    # Notebooks
    jupyterlab
    ipywidgets
    
    # Note: pip install for bleeding edge bio-ML:
    # - ESM (Meta protein model)
    # - ProGen
    # - EvoDiff
    # - RFdiffusion
    # - OpenFold
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # BIOINFORMATICS BLEEDING EDGE - SINGLE-CELL + SPATIAL
  # ════════════════════════════════════════════════════════════════════════════
  python-singlecell-bleeding = prev.python311.withPackages (ps: with ps; [
    # Single-cell core
    scanpy
    anndata
    muon
    
    # Data formats
    h5py
    zarr
    loompy
    
    # Trajectory & velocity
    scvelo
    cellrank
    
    # Integration methods
    harmonypy
    bbknn
    
    # Spatial transcriptomics
    squidpy
    
    # Deep learning for single-cell
    torch
    
    # Note: pip install for bleeding edge:
    # - scvi-tools (VAE models)
    # - cell2location
    # - CellTypist
    # - scGPT
    
    # Core
    numpy
    scipy
    pandas
    
    # Viz
    matplotlib
    seaborn
    plotly
    igraph
    leidenalg
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # BIOINFORMATICS BLEEDING EDGE - PROTEIN STRUCTURE
  # ════════════════════════════════════════════════════════════════════════════
  python-protein-bleeding = prev.python311.withPackages (ps: with ps; [
    # Structure analysis
    biopython
    mdanalysis
    
    # ML
    torch
    einops
    
    # Data
    numpy
    scipy
    pandas
    h5py
    
    # Viz
    matplotlib
    nglview
    py3dmol
    
    # Note: pip install for bleeding edge protein ML:
    # - ESMFold
    # - OpenFold
    # - RFdiffusion
    # - ProteinMPNN
    # - ColabFold
    
    # Notebooks
    jupyterlab
    ipywidgets
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # BIOINFORMATICS BLEEDING EDGE - CRISPR & GENE EDITING
  # ════════════════════════════════════════════════════════════════════════════
  python-crispr-bleeding = prev.python311.withPackages (ps: with ps; [
    # Core bioinformatics
    biopython
    pysam
    pybedtools
    
    # Primer design
    primer3-py
    
    # ML for guide prediction
    torch
    scikit-learn
    xgboost
    
    # Data
    numpy
    scipy
    pandas
    polars
    
    # Viz
    matplotlib
    seaborn
    plotly
    
    # Note: pip install for CRISPR tools:
    # - CRISPResso2 (editing analysis)
    # - Cas-OFFinder (off-target)
    # - CHOPCHOP (guide design)
    # - FlashFry (guide design)
    # - DeepCRISPR (ML prediction)
    # - CRISPR-ML
    
    # Notebooks
    jupyterlab
    ipywidgets
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # BIOINFORMATICS BLEEDING EDGE - LONG-READ SEQUENCING
  # ════════════════════════════════════════════════════════════════════════════
  python-longread-bleeding = prev.python311.withPackages (ps: with ps; [
    # Core
    biopython
    pysam
    
    # Data
    numpy
    scipy
    pandas
    h5py        # ONT fast5
    
    # ML for basecalling
    torch
    
    # Note: pip install for long-read:
    # - ont-fast5-api
    # - bonito (ONT basecaller)
    # - dorado (ONT basecaller)
    # - medaka (polishing)
    
    # Viz
    matplotlib
    seaborn
    
    # Notebooks
    jupyterlab
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # RL RESEARCH STACK
  # ════════════════════════════════════════════════════════════════════════════
  python-rl-bleeding = prev.python311.withPackages (ps: with ps; [
    # Core RL
    gymnasium
    stable-baselines3
    pettingzoo
    
    # PyTorch
    torch
    torchvision
    
    # Experiment tracking
    tensorboard
    wandb
    
    # Config
    hydra-core
    omegaconf
    
    # Viz
    matplotlib
    pygame
    opencv4
    
    # Utils
    numpy
    tqdm
    rich
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # MANIM (Mathematical Animations) - Full version
  # ════════════════════════════════════════════════════════════════════════════
  manim-full = prev.manim.override {
    # Enable all optional features
  };

  # ════════════════════════════════════════════════════════════════════════════
  # UTILITY: Bundle of all bleeding edge CLI tools
  # ════════════════════════════════════════════════════════════════════════════
  bleeding-edge-cli = prev.buildEnv {
    name = "bleeding-edge-cli";
    paths = with prev; [
      llama-cpp
      # vllm when available
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # BLEEDING EDGE BIO CLI TOOLS
  # ════════════════════════════════════════════════════════════════════════════
  bio-bleeding-cli = prev.buildEnv {
    name = "bio-bleeding-cli";
    paths = with prev; [
      # Long-read tools
      minimap2
      flye            # Long-read assembly
      
      # Fast modern tools
      seqkit
      fastp
      
      # Variant calling
      deepvariant
      
      # Note: Many cutting-edge tools require pip/conda:
      # - dorado (ONT basecaller)
      # - bonito (ONT basecaller)
      # - clair3 (variant calling)
    ];
  };
}
