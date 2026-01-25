# Overlay for scientific package customizations and fixes
# Apply via: overlays = [ science-flake.overlays.default ];

final: prev: {
  # ════════════════════════════════════════════════════════════════════════════
  # GROMACS with CUDA (pre-configured)
  # ════════════════════════════════════════════════════════════════════════════
  gromacsCuda = prev.gromacs.override {
    cudaSupport = true;
    # Uncomment and adjust for your GPU:
    # cudaArch = "86";  # RTX 30xx
    # cudaArch = "89";  # RTX 40xx
  };

  gromacs-cuda-full = prev.gromacs.override {
    cudaSupport = true;
    singlePrec = true;
    mpiEnabled = true;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # OpenBabel with all formats enabled
  # ════════════════════════════════════════════════════════════════════════════
  openbabel-full = prev.openbabel.override {
    withGUI = true;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # GDAL with all optional dependencies
  # ════════════════════════════════════════════════════════════════════════════
  gdal-full = prev.gdal.override {
    # Enable additional format support
    useHDF = true;
    useNetCDF = true;
    usePoppler = true;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Octave with GUI and common packages
  # ════════════════════════════════════════════════════════════════════════════
  octave-full = prev.octave.override {
    enableQt = true;
    enableJava = true;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Custom TeX Live with common scientific packages
  # ════════════════════════════════════════════════════════════════════════════
  texlive-science = prev.texlive.combine {
    inherit (prev.texlive)
      scheme-medium
      # Math & Science
      amsmath
      mathtools
      physics
      siunitx
      mhchem
      # Algorithms & Code
      algorithm2e
      listings
      minted
      # Diagrams
      tikz-cd
      pgfplots
      circuitikz
      # Bibliography
      biblatex
      biber
      # Utilities
      latexmk
      ;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Python with scientific stack (for quick shell access)
  # ════════════════════════════════════════════════════════════════════════════
  python-science = prev.python3.withPackages (ps: with ps; [
    numpy
    scipy
    pandas
    matplotlib
    seaborn
    scikit-learn
    jupyterlab
    sympy
    networkx
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # R with Tidyverse (for quick shell access)
  # ════════════════════════════════════════════════════════════════════════════
  r-science = prev.rWrapper.override {
    packages = with prev.rPackages; [
      tidyverse
      ggplot2
      dplyr
      tidyr
      purrr
      readr
      knitr
      rmarkdown
      devtools
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON - CRISPR & GENE EDITING STACK
  # ════════════════════════════════════════════════════════════════════════════
  python-crispr = prev.python311.withPackages (ps: with ps; [
    # Core bioinformatics
    biopython
    pysam
    pybedtools
    
    # Primer design
    primer3-py
    
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
    muon
    
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
    mdanalysis
    
    # ML for proteins
    torch
    
    # Core
    numpy
    scipy
    pandas
    
    # Visualization
    matplotlib
    nglview
    
    # Notebooks
    jupyterlab
    ipywidgets
  ]);

  # ════════════════════════════════════════════════════════════════════════════
  # PYTHON - METABOLIC ENGINEERING & SYSTEMS BIOLOGY
  # ════════════════════════════════════════════════════════════════════════════
  python-sysbio = prev.python311.withPackages (ps: with ps; [
    # Constraint-based modeling
    cobra
    
    # Systems biology
    tellurium
    
    # Data
    numpy
    scipy
    pandas
    sympy
    
    # Optimization
    cvxpy
    
    # Viz
    matplotlib
    seaborn
    networkx
    
    # Notebooks
    jupyterlab
  ]);

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
      # CRISPR analysis (if available in nixpkgs)
      # crisprVerse
      # crisprDesign
      # crisprScore
      # crisprBase
      
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
  # CLI BUNDLES - Pre-composed tool suites
  # ════════════════════════════════════════════════════════════════════════════
  bio-alignment-suite = prev.buildEnv {
    name = "bio-alignment-suite";
    paths = with prev; [
      blast diamond minimap2 bwa bowtie2 star hisat2
      mafft muscle clustal-omega jalview
    ];
  };

  bio-ngs-suite = prev.buildEnv {
    name = "bio-ngs-suite";
    paths = with prev; [
      samtools bcftools bedtools htslib
      fastqc multiqc fastp trimmomatic cutadapt
      spades freebayes snpeff
    ];
  };

  bio-rnaseq-suite = prev.buildEnv {
    name = "bio-rnaseq-suite";
    paths = with prev; [
      salmon kallisto rsem subread htseq
      star hisat2 minimap2
    ];
  };

  bio-crispr-suite = prev.buildEnv {
    name = "bio-crispr-suite";
    paths = with prev; [
      primer3 blast bwa bowtie2
      igv samtools bcftools bedtools
    ];
  };

  bio-phylo-suite = prev.buildEnv {
    name = "bio-phylo-suite";
    paths = with prev; [
      raxml iqtree mrbayes fasttree
      mafft muscle clustal-omega
      figtree trimal
    ];
  };

  bio-metagenomics-suite = prev.buildEnv {
    name = "bio-metagenomics-suite";
    paths = with prev; [
      kraken2 bracken megahit metabat2
      prokka prodigal
    ];
  };

  synbio-tools = prev.buildEnv {
    name = "synbio-tools";
    paths = with prev; [
      emboss seqkit seqtk primer3 viennarna
      graphviz jq miller csvkit
    ];
  };
}
