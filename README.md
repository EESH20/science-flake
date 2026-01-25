```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗ ██████╗██╗███████╗███╗   ██╗ ██████╗███████╗                      ║
║   ██╔════╝██╔════╝██║██╔════╝████╗  ██║██╔════╝██╔════╝                      ║
║   ███████╗██║     ██║█████╗  ██╔██╗ ██║██║     █████╗                        ║
║   ╚════██║██║     ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝                        ║
║   ███████║╚██████╗██║███████╗██║ ╚████║╚██████╗███████╗                      ║
║   ╚══════╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝                      ║
║                                                                              ║
║   ███████╗██╗      █████╗ ██╗  ██╗███████╗                                   ║
║   ██╔════╝██║     ██╔══██╗██║ ██╔╝██╔════╝                                   ║
║   █████╗  ██║     ███████║█████╔╝ █████╗                                     ║
║   ██╔══╝  ██║     ██╔══██║██╔═██╗ ██╔══╝                                     ║
║   ██║     ███████╗██║  ██║██║  ██╗███████╗                                   ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝                                   ║
║                                                                              ║
║   Production-Grade Scientific Computing & AI/ML Suite for NixOS             ║
║   ─────────────────────────────────────────────────────────────              ║
║   DeMoD LLC                                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Overview

Industry-standard development environments for:

| Domain | Templates | Key Tools |
|--------|-----------|-----------|
| **AI/ML Training** | `ml-training`, `llm`, `vision`, `audio-ml` | PyTorch, Lightning, DeepSpeed, PEFT |
| **Reinforcement Learning** | `pufferlib`, `rl` | PufferLib (1M+ steps/s), SB3, MuJoCo |
| **Scientific Computing** | `python`, `julia`, `r` | NumPy, SciPy, Tidyverse |
| **Biology & Genomics** | `bioinformatics`, `crispr`, `singlecell` | scanpy, BLAST, CRISPR tools, Bioconductor |
| **Structural Biology** | `protein`, `molecular-dynamics` | PyMOL, AlphaFold tools, GROMACS |
| **System Tools** | NixOS module | QGIS, PyMOL, LaTeX, Ollama |

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  SYSTEM LEVEL                      PROJECT LEVEL                          ┃
┃  ─────────────                     ─────────────                          ┃
┃  ┌─────────────────────┐           ┌─────────────────────────────────┐    ┃
┃  │ programs.science    │           │ nix flake init -t ...#llm       │    ┃
┃  │   .enable = true    │           │                                 │    ┃
┃  │   .cuda.enable      │           │ ┌─────────┐  ┌─────────┐        │    ┃
┃  │   .biology.crispr   │           │ │ llm/    │  │ crispr/ │  ...   │    ┃
┃  │                     │           │ │ PyTorch │  │ Primer3 │        │    ┃
┃  │ Installs:           │           │ │ vLLM    │  │ BLAST   │        │    ┃
┃  │ - llama-cpp         │           │ │ TRL     │  │ scanpy  │        │    ┃
┃  │ - QGIS, PyMOL       │           │ └─────────┘  └─────────┘        │    ┃
┃  │ - LaTeX, IGV        │           │                                 │    ┃
┃  └─────────────────────┘           │ Isolated • Reproducible • GPU   │    ┃
┃                                    └─────────────────────────────────┘    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Quick Start

### 1. Add to Flake Inputs

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    science.url = "path:/home/user/science-flake";
    science.inputs.nixpkgs.follows = "nixpkgs";
  };
}
```

### 2. Enable System Module

```nix
nixosConfigurations.myMachine = nixpkgs.lib.nixosSystem {
  modules = [
    science.nixosModules.science
    {
      programs.science = {
        enable = true;
        cuda.enable = true;
        bleedingEdge = {
          enable = true;
          llm = true;         # llama-cpp
        };
        rl.enable = true;     # MuJoCo dependencies
      };
    }
  ];
};
```

### 3. Create Project Environments

```bash
# LLM Fine-tuning
mkdir llm-project && cd llm-project
nix flake init -t /path/to/science-flake#llm
direnv allow

# CRISPR Guide Design
mkdir crispr-project && cd crispr-project
nix flake init -t /path/to/science-flake#crispr
direnv allow

# Single-Cell Analysis
mkdir singlecell-project && cd singlecell-project
nix flake init -t /path/to/science-flake#singlecell
direnv allow
```

### 4. Quick Shell Access

```bash
# Direct shell access
nix-shell shell.nix                                    # Core only
nix-shell shell.nix --arg modules '[ "core" "bio" ]'   # Multiple modules

# Interactive TUI selector (recommended)
python tui.py

# Commands available in shell:
sci-help      # Quick reference for active modules
sci-modules   # List all available modules  
sci-jupyter   # Launch JupyterLab
sci-tui       # Re-launch interactive selector
```

### 5. TUI Navigation

```
┏━━━ SCIENCE FLAKE ━━━┓
   Module Selector & Guide

▸ Core
  [✓] Core Tools         Essential data processing...
▸ Biology  
  [ ] Bioinformatics     NGS, alignment, variant calling
  [✓] CRISPR/Gene Edit   Guide design, off-target analysis
  ...

Selected (2): Core Tools, CRISPR/Gene Editing

↑↓/jk:navigate  Space:toggle  d/→:details  a:all  n:none  r:run  q:quit
```

**Keys:**
- `↑↓` or `j/k` — Navigate modules
- `Space` or `Enter` — Toggle selection  
- `d` or `→` — View module details & examples
- `a` — Select all modules
- `n` — Reset to core only
- `r` — Generate run command
- `q` — Quit

---

## Templates Reference

### AI/ML Model Building

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  TEMPLATE         DESCRIPTION                          KEY PACKAGES        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ml-training      Full training stack                  PyTorch, Lightning  ┃
┃                   Distributed, mixed precision         DeepSpeed*, NCCL    ┃
┃                                                                            ┃
┃  llm              LLM training & fine-tuning           transformers, PEFT* ┃
┃                   RLHF, DPO, inference                 TRL*, vLLM*, Ollama ┃
┃                                                                            ┃
┃  vision           Detection, segmentation, gen         torchvision, YOLO*  ┃
┃                   Multimodal                           SAM*, Diffusers*    ┃
┃                                                                            ┃
┃  audio-ml         Audio/speech/music ML                torchaudio, librosa ┃
┃                   DSP, neural synthesis                Faust, audiocraft*  ┃
┃                                                                            ┃
┃                                                   * = pip install required ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Biology & Genomics

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  TEMPLATE         DESCRIPTION                          KEY PACKAGES        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  bioinformatics   Core genomics pipeline               BLAST, samtools     ┃
┃                   Alignment, variant calling           bcftools, bedtools  ┃
┃                                                                            ┃
┃  crispr           CRISPR/gene editing design           Primer3, BLAST      ┃
┃                   Guide RNA, off-target analysis       CRISPResso2*, IGV   ┃
┃                                                                            ┃
┃  singlecell       Single-cell & spatial omics          scanpy, anndata     ┃
┃                   Trajectory, integration              scvelo, squidpy     ┃
┃                                                                            ┃
┃  rnaseq           RNA-seq quantification               STAR, salmon        ┃
┃                   Differential expression              DESeq2, kallisto    ┃
┃                                                                            ┃
┃  protein          Structural biology                   PyMOL, MDAnalysis   ┃
┃                   Protein engineering                  ESMFold*, OpenFold* ┃
┃                                                                            ┃
┃  synbio           Synthetic biology workflows          COBRApy, ViennaRNA  ┃
┃                   Metabolic engineering                EMBOSS, Primer3     ┃
┃                                                                            ┃
┃  metagenomics     Microbiome analysis                  Kraken2, MetaPhlAn  ┃
┃                   Taxonomic profiling                  MEGAHIT, prokka     ┃
┃                                                                            ┃
┃  phylogenetics    Evolutionary analysis                RAxML, IQ-TREE      ┃
┃                   Tree building & visualization        MrBayes, FigTree    ┃
┃                                                                            ┃
┃                                                   * = pip install required ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Reinforcement Learning

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  TEMPLATE         DESCRIPTION                          KEY PACKAGES        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  pufferlib        High-perf RL (1M+ steps/sec)         PufferLib*, CleanRL ┃
┃                   Ocean, Atari, NetHack, NMMO          Gymnasium, PyTorch  ┃
┃                                                                            ┃
┃  rl               General RL research                  SB3, PettingZoo     ┃
┃                   Multi-agent, MuJoCo                  MuJoCo, Ray RLlib*  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Scientific Computing

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  TEMPLATE         DESCRIPTION                          KEY PACKAGES        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  python           Data science standard                NumPy, Pandas       ┃
┃                                                        scikit-learn, Jupyter┃
┃                                                                            ┃
┃  python-gpu       GPU-accelerated ML                   PyTorch+CUDA, JAX   ┃
┃                                                                            ┃
┃  julia            Scientific Julia                     Plots, DataFrames   ┃
┃                                                        DifferentialEquations┃
┃                                                                            ┃
┃  r                Statistics                           Tidyverse, ggplot2  ┃
┃                                                        Shiny, RStudio      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Module Options

### Core

```nix
programs.science = {
  enable = true;
  
  # GPU Acceleration
  cuda = {
    enable = true;
    cudaCapabilities = [ "8.6" ];  # RTX 30xx
  };
  rocm.enable = false;  # AMD GPU (mutually exclusive with CUDA)
  
  # Core tools (always recommended)
  core.enable = true;  # pandoc, typst, gnuplot
};
```

### Biology & Genomics (NEW)

```nix
programs.science = {
  biology = {
    enable = true;           # Core bio tools (seqkit, emboss)
    
    # Workflow modules
    genomics.enable = true;        # NGS: BLAST, BWA, samtools, bcftools
    transcriptomics.enable = true; # RNA-seq: STAR, salmon, kallisto
    singleCell.enable = true;      # scanpy, anndata, scvelo, squidpy
    crispr.enable = true;          # Primer3, guide design, off-target
    structural.enable = true;      # PyMOL, Chimera, MDAnalysis
    phylogenetics.enable = true;   # RAxML, IQ-TREE, MrBayes
    metagenomics.enable = true;    # Kraken2, MetaPhlAn, MEGAHIT
    synbio.enable = true;          # COBRApy, ViennaRNA
    molecularDynamics.enable = true; # GROMACS (CUDA if enabled)
    
    # GUI applications
    gui.enable = true;       # PyMOL, IGV, Fiji, JalView
    
    # R Bioconductor
    bioconductor.enable = true;  # DESeq2, Seurat, GenomicRanges
  };
};
```

### Traditional Science

```nix
programs.science = {
  latex = {
    enable = true;
    scheme = "medium";  # minimal | small | medium | full
  };
  chemistry.enable = true;    # Avogadro, GROMACS
  physics.enable = true;      # Stellarium, ROOT
  geospatial.enable = true;   # QGIS, GDAL
  mathematics.enable = true;  # Maxima, Octave
  electronics.enable = true;  # KiCad, GNU Radio
};
```

### Bleeding Edge AI/ML

```nix
programs.science = {
  bleedingEdge = {
    enable = true;
    llm = true;           # llama-cpp, Ollama
    mlops = true;         # dvc, mlflow
    visualization = true; # manim
  };
  
  rl = {
    enable = true;
    simulation = true;    # MuJoCo dependencies
  };
};
```

---

## Biology Workflows

### CRISPR Guide Design & Analysis

```bash
cd crispr-project

# Environment includes: Primer3, BLAST, BWA, samtools, IGV

# Install specialized CRISPR tools
pip install crispresso2        # Editing analysis
pip install cas-offinder       # Off-target prediction

# Guide design workflow
python -c "
from Bio import SeqIO
from Bio.Seq import Seq
import primer3

# Load target sequence
target = 'ATGCGATCGATCGATCGATCGATCG'

# Find PAM sites (NGG for SpCas9)
pam_sites = [i for i in range(len(target)-2) 
             if target[i+1:i+3] == 'GG']

# Design primers for validation PCR
primers = primer3.design_primers(
    seq_args={'SEQUENCE_TEMPLATE': target},
    global_args={'PRIMER_PRODUCT_SIZE_RANGE': [[100, 300]]}
)
"

# Off-target analysis with BLAST
blastn -query guides.fa -db genome_db -outfmt 6 -max_target_seqs 100

# Visualize in IGV
igv -g hg38 aligned_reads.bam
```

### Single-Cell RNA-seq Analysis

```bash
cd singlecell-project

# Environment includes: scanpy, anndata, scvelo, squidpy

# Install additional tools
pip install scvi-tools         # Deep learning
pip install celltypist         # Annotation

python -c "
import scanpy as sc
import anndata as ad

# Load 10x data
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')

# Standard preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Dimensionality reduction
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Visualization
sc.pl.umap(adata, color='leiden')
"
```

### RNA-seq Differential Expression

```bash
cd rnaseq-project

# Alignment with STAR
STAR --genomeDir star_index \
     --readFilesIn reads_R1.fq.gz reads_R2.fq.gz \
     --readFilesCommand zcat \
     --outSAMtype BAM SortedByCoordinate

# Quantification with salmon
salmon quant -i salmon_index -l A \
       -1 reads_R1.fq.gz -2 reads_R2.fq.gz \
       -o quants/sample1

# R analysis with DESeq2
Rscript -e "
library(DESeq2)
library(tximport)

# Import salmon counts
files <- list.files('quants', pattern='quant.sf', recursive=TRUE, full.names=TRUE)
txi <- tximport(files, type='salmon', tx2gene=tx2gene)

# DESeq2 analysis
dds <- DESeqDataSetFromTximport(txi, colData, ~condition)
dds <- DESeq(dds)
res <- results(dds, contrast=c('condition', 'treated', 'control'))
"
```

### Protein Structure Prediction

```bash
cd protein-project

# Install bleeding-edge protein ML
pip install esm                # Meta's ESM
pip install openfold           # OpenFold (requires setup)

# ESMFold prediction
python -c "
import torch
import esm

# Load ESMFold
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Predict structure
sequence = 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
with torch.no_grad():
    output = model.infer_pdb(sequence)

# Save PDB
with open('predicted.pdb', 'w') as f:
    f.write(output)
"

# Visualize with PyMOL
pymol predicted.pdb
```

### Molecular Dynamics with GROMACS

```bash
cd md-project

# Prepare system
gmx pdb2gmx -f protein.pdb -o protein.gro -water spce
gmx editconf -f protein.gro -o box.gro -c -d 1.0 -bt cubic
gmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top

# Add ions
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr
gmx genion -s ions.tpr -o system.gro -p topol.top -pname NA -nname CL -neutral

# Energy minimization
gmx grompp -f em.mdp -c system.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# Production MD (GPU accelerated if CUDA enabled)
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
gmx mdrun -deffnm md -nb gpu
```

---

## Directory Structure

```
science-flake/
├── flake.nix                 # Root flake with module exports
├── shell.nix                 # Unified dev shell with module selection
├── tui.py                    # Interactive curses TUI selector
├── README.md                 # This file
│
├── modules/
│   ├── science.nix           # NixOS module (system-level)
│   └── science-hm.nix        # Home Manager module
│
├── overlays/
│   ├── science.nix           # Package customizations
│   ├── biology.nix           # Biology-specific overlays (NEW)
│   └── bleeding-edge.nix     # Latest AI/ML packages
│
└── templates/
    ├── python/               # Standard data science
    ├── python-gpu/           # GPU-accelerated ML
    ├── julia/                # Scientific Julia
    ├── r/                    # R statistics
    │
    ├── ml-training/          # Full ML training stack
    ├── llm/                  # LLM development
    ├── vision/               # Computer vision
    ├── audio-ml/             # Audio/DSP ML
    │
    ├── pufferlib/            # High-perf RL
    ├── rl/                   # General RL research
    │
    ├── bioinformatics/       # Core genomics
    ├── crispr/               # CRISPR/gene editing (NEW)
    ├── singlecell/           # Single-cell omics (NEW)
    ├── rnaseq/               # RNA-seq (NEW)
    ├── protein/              # Structural biology (NEW)
    ├── synbio/               # Synthetic biology (NEW)
    ├── metagenomics/         # Microbiome (NEW)
    └── phylogenetics/        # Evolutionary (NEW)
```

---

## GPU Configuration

### NVIDIA CUDA

```nix
# In your NixOS configuration
hardware.nvidia = {
  package = config.boot.kernelPackages.nvidiaPackages.stable;
  modesetting.enable = true;
  open = false;  # or true for open-source driver (RTX 20xx+)
};

programs.science.cuda = {
  enable = true;
  cudaCapabilities = [ "8.6" ];  # Match your GPU
};
```

| GPU Series | Capability |
|------------|------------|
| RTX 20xx   | 7.5        |
| RTX 30xx   | 8.6        |
| RTX 40xx   | 8.9        |
| H100       | 9.0        |

### AMD ROCm

```nix
hardware.amdgpu.opencl.enable = true;
programs.science.rocm.enable = true;
```

---

## Overlays

```nix
nixpkgs.overlays = [
  science.overlays.default       # Package customizations
  science.overlays.biology       # Biology-specific (NEW)
  science.overlays.bleedingEdge  # Latest AI/ML
  science.overlays.unstable      # nixpkgs-unstable access
];
```

Available packages from overlays:

**ML/AI:**
- `python-ml-bleeding` - Full ML stack
- `python-rl-bleeding` - RL research stack  
- `llama-cpp-cuda` - GPU-accelerated inference

**Biology:**
- `python-crispr` - CRISPR design & analysis
- `python-singlecell` - Single-cell genomics
- `python-structural` - Protein structure
- `python-sysbio` - Systems biology
- `python-bioml-bleeding` - ML for biology
- `r-bioconductor` - Bioconductor packages
- `r-crispr` - CRISPR R packages
- `bio-alignment-suite` - CLI alignment tools
- `bio-ngs-suite` - NGS processing
- `bio-crispr-suite` - CRISPR CLI tools
- `bio-phylo-suite` - Phylogenetics
- `synbio-tools` - Synthetic biology

**Chemistry/MD:**
- `gromacsCuda` - GPU-accelerated molecular dynamics
- `gromacs-cuda-full` - Full GROMACS + CUDA + MPI

---

## License

MIT

---

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                            ┃
┃   DeMoD LLC                                                                ┃
┃   "Build different."                                                       ┃
┃                                                                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```
