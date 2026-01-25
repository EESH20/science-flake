# Biology & Genetic Engineering Overlay
# Apply via: overlays = [ science-flake.overlays.biology ];
#
# Comprehensive tools for molecular biology, genomics, CRISPR design,
# and synthetic biology workflows.

final: prev: {
  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON - CRISPR & GENE EDITING STACK
  # ════════════════════════════════════════════════════════════════════════════
  python-crispr = prev.python311.withPackages (ps: with ps; [
    # Core bioinformatics
    biopython
    pysam
    pybedtools
    
    # Sequence analysis
    primer3-py      # Primer design
    # Note: CRISPResso2, Cas-OFFinder typically pip-installed
    
    # Structural biology
    # biotite       # Sequence/structure analysis
    
    # Data processing
    numpy
    scipy
    pandas
    polars
    
    # Visualization
    matplotlib
    seaborn
    plotly
    
    # Notebooks
    jupyterlab
    ipywidgets
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON - SINGLE-CELL & SPATIAL GENOMICS
  # ════════════════════════════════════════════════════════════════════════════
  python-singlecell = prev.python311.withPackages (ps: with ps; [
    # Single-cell core
    scanpy
    anndata
    muon            # Multi-omics
    
    # Data formats
    h5py
    zarr
    loompy
    
    # Trajectory & RNA velocity
    scvelo
    cellrank
    
    # Integration
    harmonypy
    bbknn
    
    # Spatial
    squidpy
    
    # ML for biology
    torch
    scikit-learn
    
    # Core
    numpy
    scipy
    pandas
    
    # Viz
    matplotlib
    seaborn
    plotly
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON - STRUCTURAL BIOLOGY & PROTEIN ENGINEERING
  # ════════════════════════════════════════════════════════════════════════════
  python-structural = prev.python311.withPackages (ps: with ps; [
    # Structure analysis
    biopython
    mdanalysis      # Molecular dynamics analysis
    
    # ML for proteins
    torch
    
    # Core
    numpy
    scipy
    pandas
    
    # Visualization
    matplotlib
    nglview         # 3D molecular viz in notebooks
    
    # Notebooks
    jupyterlab
    ipywidgets
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON - METABOLIC ENGINEERING & SYSTEMS BIOLOGY
  # ════════════════════════════════════════════════════════════════════════════
  python-sysbio = prev.python311.withPackages (ps: with ps; [
    # Constraint-based modeling
    cobra           # COBRApy - metabolic modeling
    
    # Systems biology
    tellurium       # Simulation
    
    # Data
    numpy
    scipy
    pandas
    sympy
    
    # Optimization
    scipy
    cvxpy
    
    # Viz
    matplotlib
    seaborn
    networkx
    
    # Notebooks
    jupyterlab
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # CLI TOOLS - SEQUENCE ALIGNMENT & ANALYSIS
  # ════════════════════════════════════════════════════════════════════════════
  bio-alignment-suite = prev.buildEnv {
    name = "bio-alignment-suite";
    paths = with prev; [
      # Sequence alignment
      blast           # NCBI BLAST+
      diamond         # Fast protein alignment
      minimap2        # Long-read alignment
      bwa             # Short-read alignment
      bowtie2         # Short-read alignment
      star            # RNA-seq alignment
      hisat2          # RNA-seq alignment
      
      # Multiple sequence alignment
      mafft           # Fast MSA
      muscle          # Accurate MSA
      clustal-omega   # Clustal Omega
      
      # Alignment viewers
      jalview         # MSA visualization
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # CLI TOOLS - NGS DATA PROCESSING
  # ════════════════════════════════════════════════════════════════════════════
  bio-ngs-suite = prev.buildEnv {
    name = "bio-ngs-suite";
    paths = with prev; [
      # Core NGS tools
      samtools        # SAM/BAM manipulation
      bcftools        # VCF/BCF manipulation
      bedtools        # BED manipulation
      htslib          # High-throughput sequencing lib
      
      # Quality control
      fastqc          # QC reports
      multiqc         # Aggregate QC
      fastp           # Fast preprocessing
      trimmomatic     # Read trimming
      cutadapt        # Adapter trimming
      
      # Assembly
      spades          # Genome assembly
      megahit         # Metagenome assembly
      flye            # Long-read assembly
      
      # Variant calling
      freebayes       # Variant caller
      gatk            # GATK toolkit
      deepvariant     # ML variant caller
      
      # Annotation
      snpeff          # Variant annotation
      vep             # Variant Effect Predictor
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # CLI TOOLS - RNA-SEQ & TRANSCRIPTOMICS
  # ════════════════════════════════════════════════════════════════════════════
  bio-rnaseq-suite = prev.buildEnv {
    name = "bio-rnaseq-suite";
    paths = with prev; [
      # Quantification
      salmon          # Fast transcript quant
      kallisto        # Pseudoalignment quant
      rsem            # RNA-seq quant
      featurecounts   # Read counting
      htseq           # Read counting
      
      # Differential expression (R-based)
      # DESeq2, edgeR handled via R environment
      
      # Alignment
      star            # RNA-seq aligner
      hisat2          # Splice-aware aligner
      
      # Long-read RNA
      minimap2        # Long-read alignment
      isoquant        # Isoform quant
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # CLI TOOLS - CRISPR & GENOME EDITING
  # ════════════════════════════════════════════════════════════════════════════
  bio-crispr-suite = prev.buildEnv {
    name = "bio-crispr-suite";
    paths = with prev; [
      # Guide RNA design (supporting tools)
      primer3         # Primer/oligo design
      
      # Sequence analysis
      blast           # Off-target search
      bwa             # Alignment
      bowtie2         # Alignment
      
      # Visualization
      igv             # Genome browser
      
      # Data processing
      samtools
      bcftools
      bedtools
      
      # Note: Primary CRISPR tools typically pip-installed:
      # - CRISPResso2 (editing analysis)
      # - Cas-OFFinder (off-target prediction)
      # - CHOPCHOP (guide design)
      # - FlashFry (guide design)
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # GUI APPLICATIONS - MOLECULAR BIOLOGY
  # ════════════════════════════════════════════════════════════════════════════
  bio-gui-suite = prev.buildEnv {
    name = "bio-gui-suite";
    paths = with prev; [
      # Structure visualization
      pymol           # Molecular graphics
      chimera         # UCSF Chimera
      vmd             # Visual Molecular Dynamics
      
      # Sequence visualization
      igv             # Integrative Genomics Viewer
      jalview         # Alignment viewer
      ugene           # Integrated bioinfo workbench
      
      # Image analysis
      fiji            # ImageJ distribution
      cellprofiler    # Cell image analysis
      
      # Cloning/plasmid design
      # ApE, Benchling, SnapGene are commercial/web-based
      # Open source alternatives:
      # - pDraw (wine)
      # - Genome Compiler (web)
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # R - BIOCONDUCTOR STACK
  # ════════════════════════════════════════════════════════════════════════════
  r-bioconductor = prev.rWrapper.override {
    packages = with prev.rPackages; [
      # Core Bioconductor
      BiocManager
      Biostrings
      GenomicRanges
      GenomicFeatures
      GenomicAlignments
      Rsamtools
      VariantAnnotation
      
      # RNA-seq
      DESeq2
      edgeR
      limma
      tximport
      
      # Single-cell
      Seurat
      SingleCellExperiment
      scater
      scran
      
      # Annotation
      AnnotationDbi
      org_Hs_eg_db
      org_Mm_eg_db
      GO_db
      
      # Visualization
      ggplot2
      ComplexHeatmap
      EnhancedVolcano
      clusterProfiler
      
      # Utilities
      tidyverse
      data_table
      BiocParallel
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # R - CRISPR ANALYSIS
  # ════════════════════════════════════════════════════════════════════════════
  r-crispr = prev.rWrapper.override {
    packages = with prev.rPackages; [
      # CRISPR analysis
      crisprVerse      # CRISPR guide design ecosystem
      crisprDesign     # Guide RNA design
      crisprScore      # Scoring algorithms
      crisprBase       # Core functions
      
      # Supporting packages
      Biostrings
      GenomicRanges
      BSgenome_Hsapiens_UCSC_hg38
      BSgenome_Mmusculus_UCSC_mm10
      
      # Utilities
      tidyverse
      ggplot2
      data_table
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # JULIA - BIOLOGY STACK
  # ════════════════════════════════════════════════════════════════════════════
  julia-bio = prev.julia.withPackages [
    "BioSequences"
    "BioAlignments"
    "FASTX"
    "GenomicFeatures"
    "XAM"             # SAM/BAM
    "VariantCallFormat"
    "Plots"
    "DataFrames"
  ];

  # ════════════════════════════════════════════════════════════════════════════
  # MOLECULAR DYNAMICS
  # ════════════════════════════════════════════════════════════════════════════
  gromacs-full = prev.gromacs.override {
    singlePrec = true;
    mpiEnabled = true;
  };

  gromacs-cuda-full = prev.gromacs.override {
    cudaSupport = true;
    singlePrec = true;
    mpiEnabled = true;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # SYNTHETIC BIOLOGY WORKFLOW
  # ════════════════════════════════════════════════════════════════════════════
  synbio-tools = prev.buildEnv {
    name = "synbio-tools";
    paths = with prev; [
      # Sequence manipulation
      emboss          # EMBOSS suite
      seqkit          # FASTA/Q toolkit
      seqtk           # Sequence processing
      
      # Primer design
      primer3
      
      # Structure
      viennarna       # RNA secondary structure
      
      # Visualization
      graphviz
      
      # Data
      jq
      miller
      csvkit
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PHYLOGENETICS & EVOLUTION
  # ════════════════════════════════════════════════════════════════════════════
  bio-phylo-suite = prev.buildEnv {
    name = "bio-phylo-suite";
    paths = with prev; [
      # Tree building
      raxml           # Maximum likelihood
      iqtree          # ML phylogenetics
      mrbayes         # Bayesian inference
      fasttree        # Fast approximate ML
      
      # Tree visualization
      figtree         # Tree viewer
      
      # MSA (required for phylo)
      mafft
      muscle
      clustal-omega
      
      # Utilities
      trimal          # Alignment trimming
      modeltest-ng    # Model selection
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # METAGENOMICS & MICROBIOME
  # ════════════════════════════════════════════════════════════════════════════
  bio-metagenomics-suite = prev.buildEnv {
    name = "bio-metagenomics-suite";
    paths = with prev; [
      # Taxonomic profiling
      kraken2         # Taxonomic classification
      bracken         # Abundance estimation
      metaphlan       # MetaPhlAn
      
      # Assembly
      megahit         # Metagenome assembly
      metaspades      # SPAdes for metagenomes
      
      # Binning
      maxbin2         # Genome binning
      metabat2        # Metagenome binning
      
      # Quality & stats
      checkm          # Genome completeness
      
      # Functional
      prokka          # Annotation
      prodigal        # Gene prediction
    ];
  };
}
