# Science Flake - Unified Development Shell
# Version: 1.0.0
# ══════════════════════════════════════════════════════════════════════════════
#
# USAGE:
#   nix-shell shell.nix                                    # Core only
#   nix-shell shell.nix --arg modules '[ "core" "bio" ]'   # Multiple modules
#   python tui.py                                          # Interactive selector
#
# MODULES: core bio crispr singlecell rnaseq protein synbio ml rl geo chem
#
# ══════════════════════════════════════════════════════════════════════════════

{ pkgs ? import <nixpkgs> {}
, modules ? [ "core" ]
}:

let
  version = "1.0.0";
  
  # Valid module names for validation
  validModules = [ "core" "bio" "crispr" "singlecell" "rnaseq" "protein" "synbio" "ml" "rl" "geo" "chem" ];
  
  # Validate modules
  invalidModules = builtins.filter (m: !(builtins.elem m validModules)) modules;
  
  # Ensure core is always included
  finalModules = if builtins.elem "core" modules then modules else [ "core" ] ++ modules;
  # ══════════════════════════════════════════════════════════════════════════
  # COLOR DEFINITIONS (ANSI escape codes)
  # ══════════════════════════════════════════════════════════════════════════
  c = {
    reset   = "\\033[0m";
    bold    = "\\033[1m";
    dim     = "\\033[2m";
    red     = "\\033[31m";
    green   = "\\033[32m";
    yellow  = "\\033[33m";
    blue    = "\\033[34m";
    magenta = "\\033[35m";
    cyan    = "\\033[36m";
    white   = "\\033[37m";
  };

  # ══════════════════════════════════════════════════════════════════════════
  # PACKAGE MODULES
  # ══════════════════════════════════════════════════════════════════════════
  modulePackages = {
    core = with pkgs; [
      jq yq-go csvkit miller
      pandoc typst
      gnuplot graphviz
      (python311.withPackages (ps: with ps; [
        numpy scipy pandas matplotlib seaborn
        jupyterlab ipywidgets requests rich
      ]))
    ];

    bio = with pkgs; [
      seqkit seqtk emboss
      blast bwa bowtie2 minimap2 diamond
      mafft muscle clustal-omega
      samtools bcftools bedtools htslib
      fastqc multiqc fastp
      igv jalview
      (python311.withPackages (ps: with ps; [
        biopython pysam pybedtools
        numpy scipy pandas matplotlib seaborn jupyterlab
      ]))
    ];

    crispr = with pkgs; [
      primer3
      blast bwa bowtie2
      samtools bcftools bedtools
      igv
      (python311.withPackages (ps: with ps; [
        biopython pysam pybedtools primer3-py
        numpy scipy pandas polars
        matplotlib seaborn plotly
        scikit-learn xgboost
        jupyterlab ipywidgets
      ]))
    ];

    singlecell = with pkgs; [
      (python311.withPackages (ps: with ps; [
        scanpy anndata muon
        scvelo cellrank squidpy
        harmonypy bbknn leidenalg igraph
        h5py zarr loompy
        torch scikit-learn
        numpy scipy pandas
        matplotlib seaborn plotly
        jupyterlab ipywidgets
      ]))
    ];

    rnaseq = with pkgs; [
      star hisat2
      salmon kallisto rsem subread htseq
      fastqc multiqc rseqc qualimap
      (rWrapper.override {
        packages = with rPackages; [
          DESeq2 edgeR limma tximport
          ggplot2 pheatmap EnhancedVolcano
          tidyverse data_table
        ];
      })
    ];

    protein = with pkgs; [
      pymol chimera vmd viennarna
      (python311.withPackages (ps: with ps; [
        biopython mdanalysis
        torch einops
        numpy scipy pandas h5py
        matplotlib nglview
        jupyterlab ipywidgets
      ]))
    ];

    synbio = with pkgs; [
      emboss seqkit seqtk primer3 viennarna graphviz
      (python311.withPackages (ps: with ps; [
        biopython cobra tellurium
        numpy scipy pandas sympy
        networkx cvxpy
        matplotlib seaborn
        jupyterlab
      ]))
    ];

    ml = with pkgs; [
      (python311.withPackages (ps: with ps; [
        torch torchvision torchaudio
        transformers datasets tokenizers accelerate safetensors
        scikit-learn xgboost lightgbm
        numpy scipy pandas polars
        matplotlib seaborn plotly
        tensorboard wandb
        jupyterlab ipywidgets
      ]))
      llama-cpp
    ];

    rl = with pkgs; [
      (python311.withPackages (ps: with ps; [
        gymnasium stable-baselines3 pettingzoo
        torch torchvision
        tensorboard wandb
        hydra-core omegaconf
        matplotlib pygame opencv4
        numpy tqdm rich
        jupyterlab
      ]))
    ];

    geo = with pkgs; [
      qgis grass gdal proj geos
      (python311.withPackages (ps: with ps; [
        geopandas rasterio fiona shapely pyproj
        numpy pandas matplotlib jupyterlab
      ]))
    ];

    chem = with pkgs; [
      avogadro2 openbabel jmol gromacs
      (python311.withPackages (ps: with ps; [
        rdkit numpy scipy pandas matplotlib jupyterlab
      ]))
    ];
  };

  # ══════════════════════════════════════════════════════════════════════════
  # HELP CONTENT (structured for easy maintenance)
  # ══════════════════════════════════════════════════════════════════════════
  helpContent = {
    core = {
      title = "CORE TOOLS";
      tools = "jq yq csvkit miller pandoc typst gnuplot";
      python = "numpy scipy pandas matplotlib jupyter";
      examples = [
        "mlr --icsv --opprint cat data.csv    # View CSV"
        "jq '.[] | .name' data.json           # Parse JSON"
        "pandoc doc.md -o doc.pdf             # Convert"
      ];
    };
    bio = {
      title = "BIOINFORMATICS";
      tools = "blast bwa bowtie2 minimap2 samtools bcftools bedtools fastqc";
      python = "biopython pysam pybedtools";
      examples = [
        "seqkit stats *.fastq.gz              # FASTQ stats"
        "bwa mem ref.fa R1.fq R2.fq | samtools sort -o out.bam"
        "bcftools stats variants.vcf          # VCF stats"
      ];
    };
    crispr = {
      title = "CRISPR/GENE EDITING";
      tools = "primer3 blast bwa bowtie2 igv";
      python = "biopython primer3-py pysam";
      pip = "crispresso2 chopchop";
      examples = [
        "seqkit locate -p '.GG' target.fa     # Find PAM (SpCas9)"
        "blastn -query guides.fa -db genome   # Off-target"
      ];
    };
    singlecell = {
      title = "SINGLE-CELL & SPATIAL";
      tools = "(Python-based)";
      python = "scanpy anndata muon scvelo squidpy";
      pip = "scvi-tools celltypist";
      examples = [
        "adata = sc.read_10x_mtx('matrix/')   # Load 10x"
        "sc.pp.neighbors(adata); sc.tl.umap(adata)"
      ];
    };
    rnaseq = {
      title = "RNA-SEQ";
      tools = "star hisat2 salmon kallisto rsem";
      r = "DESeq2 edgeR limma tximport";
      examples = [
        "salmon quant -i idx -l A -1 R1.fq -2 R2.fq -o quant"
        "# R: res <- results(DESeq(dds))"
      ];
    };
    protein = {
      title = "STRUCTURAL BIOLOGY";
      tools = "pymol chimera vmd viennarna";
      python = "biopython mdanalysis nglview";
      pip = "esm openfold";
      examples = [
        "pymol structure.pdb                  # Visualize"
        "echo 'GCGCUUCGAGCG' | RNAfold        # RNA fold"
      ];
    };
    synbio = {
      title = "SYNTHETIC BIOLOGY";
      tools = "emboss seqkit primer3 viennarna";
      python = "cobra tellurium biopython";
      examples = [
        "model = cobra.io.read_sbml_model('model.xml')"
        "seqkit seq -rp sequence.fa           # RevComp"
      ];
    };
    ml = {
      title = "MACHINE LEARNING";
      tools = "llama-cpp";
      python = "torch transformers scikit-learn xgboost wandb";
      pip = "peft trl bitsandbytes flash-attn";
      examples = [
        "from transformers import AutoModelForCausalLM"
        "llama-cli -m model.gguf -p 'Hello'"
      ];
    };
    rl = {
      title = "REINFORCEMENT LEARNING";
      tools = "(Python-based)";
      python = "gymnasium stable-baselines3 pettingzoo torch";
      pip = "pufferlib mujoco";
      examples = [
        "model = PPO('MlpPolicy', 'CartPole-v1')"
        "model.learn(total_timesteps=10000)"
      ];
    };
    geo = {
      title = "GEOSPATIAL";
      tools = "qgis grass gdal proj geos";
      python = "geopandas rasterio fiona shapely";
      examples = [
        "gdal_translate -of GTiff in.img out.tif"
        "gdf = gpd.read_file('data.shp')"
      ];
    };
    chem = {
      title = "CHEMISTRY";
      tools = "avogadro2 openbabel jmol gromacs";
      python = "rdkit";
      examples = [
        "obabel mol.sdf -O mol.pdb             # Convert"
        "gmx pdb2gmx -f protein.pdb            # GROMACS"
      ];
    };
  };

  # ══════════════════════════════════════════════════════════════════════════
  # HELP TEXT GENERATOR
  # ══════════════════════════════════════════════════════════════════════════
  formatHelp = name: let
    h = helpContent.${name};
    pipLine = if h ? pip then "  ${c.yellow}pip:${c.reset}     ${h.pip}\n" else "";
    rLine = if h ? r then "  ${c.magenta}R:${c.reset}       ${h.r}\n" else "";
    exampleLines = builtins.concatStringsSep "\n" (map (e: "  ${c.green}${e}${c.reset}") h.examples);
  in ''
    ${c.cyan}━━━ ${h.title} ━━━${c.reset}
    ${c.bold}tools:${c.reset}   ${h.tools}
    ${c.bold}python:${c.reset}  ${h.python or ""}
    ${rLine}${pipLine}
    ${exampleLines}
  '';

  selectedHelp = builtins.concatStringsSep "\n" (
    map formatHelp (builtins.filter (m: builtins.hasAttr m helpContent) finalModules)
  );

  moduleList = builtins.concatStringsSep " " finalModules;
  moduleCount = builtins.length finalModules;

  # ══════════════════════════════════════════════════════════════════════════
  # BUILD PACKAGE LIST
  # ══════════════════════════════════════════════════════════════════════════
  selectedPackages = builtins.concatLists (
    builtins.map (m: modulePackages.${m} or []) finalModules
  );

  # ══════════════════════════════════════════════════════════════════════════
  # SHELL SCRIPTS
  # ══════════════════════════════════════════════════════════════════════════
  helpScript = pkgs.writeShellScriptBin "sci-help" ''
    echo -e ""
    echo -e "${c.bold}${c.blue}╔═══════════════════════════════════════════════════════════════╗${c.reset}"
    echo -e "${c.bold}${c.blue}║${c.reset}  ${c.cyan}${c.bold}SCIENCE FLAKE${c.reset} ${c.dim}v${version}${c.reset}                                      ${c.bold}${c.blue}║${c.reset}"
    echo -e "${c.bold}${c.blue}╚═══════════════════════════════════════════════════════════════╝${c.reset}"
    echo -e ""
    echo -e "${c.dim}Active:${c.reset} ${c.yellow}${moduleList}${c.reset} (${moduleCount} modules)"
    echo -e ""
    echo -e "${selectedHelp}"
    echo -e ""
    echo -e "${c.dim}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${c.reset}"
    echo -e "${c.bold}sci-help${c.reset}     This reference"
    echo -e "${c.bold}sci-modules${c.reset}  List all modules"
    echo -e "${c.bold}sci-jupyter${c.reset}  Launch JupyterLab"
    echo -e "${c.bold}sci-tui${c.reset}      Interactive selector"
    echo -e "${c.bold}sci-info${c.reset}     Show environment info"
    echo -e ""
  '';

  infoScript = pkgs.writeShellScriptBin "sci-info" ''
    echo -e ""
    echo -e "${c.bold}${c.cyan}Environment Info${c.reset}"
    echo -e "${c.dim}─────────────────────────────────────────────────${c.reset}"
    echo -e "${c.bold}Version:${c.reset}  ${version}"
    echo -e "${c.bold}Modules:${c.reset}  ${moduleList}"
    echo -e "${c.bold}Python:${c.reset}   $(python --version 2>&1)"
    echo -e "${c.bold}Shell:${c.reset}    $SHELL"
    echo -e ""
    echo -e "${c.dim}Package counts by module:${c.reset}"
    echo -e "  core: 12  bio: 16  crispr: 10  singlecell: 15"
    echo -e "  rnaseq: 12  protein: 8  synbio: 10  ml: 14"
    echo -e "  rl: 10  geo: 10  chem: 8"
    echo -e ""
  '';

  modulesScript = pkgs.writeShellScriptBin "sci-modules" ''
    echo -e ""
    echo -e "${c.bold}${c.cyan}Available Modules${c.reset}"
    echo -e "${c.dim}─────────────────────────────────────────────────${c.reset}"
    echo -e ""
    echo -e "${c.yellow}Core${c.reset}"
    echo -e "  ${c.green}core${c.reset}        Data processing, pandas, jupyter"
    echo -e ""
    echo -e "${c.yellow}Biology${c.reset}"
    echo -e "  ${c.green}bio${c.reset}         NGS, alignment, variant calling"
    echo -e "  ${c.green}crispr${c.reset}      Guide design, off-target analysis"
    echo -e "  ${c.green}singlecell${c.reset}  scRNA-seq, spatial (scanpy)"
    echo -e "  ${c.green}rnaseq${c.reset}      Bulk RNA-seq (STAR, DESeq2)"
    echo -e "  ${c.green}protein${c.reset}     Structure (PyMOL, MDAnalysis)"
    echo -e "  ${c.green}synbio${c.reset}      Metabolic modeling (COBRApy)"
    echo -e ""
    echo -e "${c.yellow}AI/ML${c.reset}"
    echo -e "  ${c.green}ml${c.reset}          PyTorch, transformers, llama.cpp"
    echo -e "  ${c.green}rl${c.reset}          Gymnasium, Stable-Baselines3"
    echo -e ""
    echo -e "${c.yellow}Science${c.reset}"
    echo -e "  ${c.green}geo${c.reset}         QGIS, geopandas, GDAL"
    echo -e "  ${c.green}chem${c.reset}        GROMACS, RDKit, OpenBabel"
    echo -e ""
    echo -e "${c.dim}─────────────────────────────────────────────────${c.reset}"
    echo -e "${c.bold}Usage:${c.reset}"
    echo -e "  nix-shell shell.nix --arg modules '[ \"core\" \"bio\" ]'"
    echo -e "  ${c.dim}or use${c.reset} sci-tui ${c.dim}for interactive selection${c.reset}"
    echo -e ""
  '';

  jupyterScript = pkgs.writeShellScriptBin "sci-jupyter" ''
    echo -e "${c.cyan}Starting JupyterLab...${c.reset}"
    echo -e "${c.dim}Press Ctrl+C to stop${c.reset}"
    jupyter lab --no-browser
  '';

  tuiScript = pkgs.writeShellScriptBin "sci-tui" ''
    if [ -f ./tui.py ]; then
      python ./tui.py
    elif [ -f $HOME/science-flake/tui.py ]; then
      python $HOME/science-flake/tui.py
    else
      echo -e "${c.red}TUI not found.${c.reset} Run from science-flake directory or install tui.py"
      echo -e "Get it: curl -O https://raw.githubusercontent.com/.../tui.py"
    fi
  '';

  # ══════════════════════════════════════════════════════════════════════════
  # BANNER
  # ══════════════════════════════════════════════════════════════════════════
  banner = ''
    echo -e ""
    echo -e "${c.blue}${c.bold}  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${c.reset}"
    echo -e "${c.blue}${c.bold}  ┃${c.reset}  ${c.cyan}${c.bold}SCIENCE FLAKE${c.reset} ${c.dim}v${version}${c.reset}                              ${c.blue}${c.bold}┃${c.reset}"
    echo -e "${c.blue}${c.bold}  ┃${c.reset}  ${c.dim}Production Scientific Computing${c.reset}                       ${c.blue}${c.bold}┃${c.reset}"
    echo -e "${c.blue}${c.bold}  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${c.reset}"
    echo -e ""
    echo -e "  ${c.dim}Modules:${c.reset} ${c.yellow}${moduleList}${c.reset} ${c.dim}(${moduleCount})${c.reset}"
    echo -e "  ${c.dim}Commands:${c.reset} ${c.green}sci-help${c.reset} ${c.dim}│${c.reset} ${c.green}sci-modules${c.reset} ${c.dim}│${c.reset} ${c.green}sci-jupyter${c.reset} ${c.dim}│${c.reset} ${c.green}sci-tui${c.reset}"
    echo -e ""
  '';

in 
  # Validate modules before building
  assert (invalidModules == []) || builtins.throw 
    "Invalid modules: ${builtins.concatStringsSep ", " invalidModules}. Valid: ${builtins.concatStringsSep ", " validModules}";
  
pkgs.mkShell {
  name = "science-shell-${version}";
  
  packages = selectedPackages ++ [
    helpScript
    modulesScript
    jupyterScript
    tuiScript
    infoScript
  ];

  shellHook = banner;

  # Environment
  PYTHONDONTWRITEBYTECODE = "1";
  PYTHONUNBUFFERED = "1";
  JUPYTER_CONFIG_DIR = "$HOME/.jupyter";
  BLASTDB = "$HOME/.local/share/blast/db";
}
