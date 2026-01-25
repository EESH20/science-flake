{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.programs.science;
  
  cudaPkg = pkg: cudaPkg: if cfg.cuda.enable then cudaPkg else pkg;
  
in {
  # ════════════════════════════════════════════════════════════════════════════
  # OPTIONS (Simplified for Home Manager - user-level packages only)
  # ════════════════════════════════════════════════════════════════════════════
  options.programs.science = {
    enable = mkEnableOption "Scientific Computing Suite (User-level)";

    cuda.enable = mkEnableOption "CUDA GPU support (requires system NVIDIA drivers)";
    rocm.enable = mkEnableOption "ROCm GPU support (requires system AMD drivers)";

    core.enable = mkOption {
      type = types.bool;
      default = true;
      description = "Core tools (Pandoc, Typst, gnuplot)";
    };

    latex = {
      enable = mkEnableOption "LaTeX publishing suite";
      scheme = mkOption {
        type = types.enum [ "minimal" "small" "medium" "full" ];
        default = "medium";
        description = "TeX Live distribution size";
      };
    };

    # ══════════════════════════════════════════════════════════════════════════
    # BIOLOGY - Comprehensive Options
    # ══════════════════════════════════════════════════════════════════════════
    biology = {
      enable = mkEnableOption "Biology tools (core bioinformatics suite)";
      
      # Workflow-specific modules
      genomics = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "NGS analysis (alignment, variant calling, QC)";
        };
      };
      
      transcriptomics = {
        enable = mkEnableOption "RNA-seq and transcriptomics tools";
      };
      
      singleCell = {
        enable = mkEnableOption "Single-cell and spatial genomics";
      };
      
      crispr = {
        enable = mkEnableOption "CRISPR/gene editing design and analysis tools";
      };
      
      structural = {
        enable = mkEnableOption "Structural biology and protein analysis";
      };
      
      phylogenetics = {
        enable = mkEnableOption "Phylogenetics and evolutionary analysis";
      };
      
      metagenomics = {
        enable = mkEnableOption "Metagenomics and microbiome analysis";
      };
      
      synbio = {
        enable = mkEnableOption "Synthetic biology workflow tools";
      };
      
      molecularDynamics = {
        enable = mkEnableOption "Molecular dynamics (GROMACS, etc.)";
      };
      
      # GUI applications
      gui = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "GUI applications (PyMOL, IGV, Fiji)";
        };
      };
      
      # R Bioconductor
      bioconductor = {
        enable = mkEnableOption "R with Bioconductor packages";
      };
    };

    chemistry.enable = mkEnableOption "Chemistry tools (Avogadro, OpenBabel)";
    physics.enable = mkEnableOption "Physics tools (Stellarium)";
    geospatial.enable = mkEnableOption "GIS tools (QGIS, GDAL)";
    mathematics.enable = mkEnableOption "Math tools (Maxima, Octave)";
    electronics.enable = mkEnableOption "Electronics tools (KiCad)";

    extraPackages = mkOption {
      type = types.listOf types.package;
      default = [];
      description = "Additional packages";
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # CONFIG
  # ════════════════════════════════════════════════════════════════════════════
  config = mkIf cfg.enable {
    assertions = [{
      assertion = !(cfg.cuda.enable && cfg.rocm.enable);
      message = "Cannot enable both CUDA and ROCm";
    }];

    home.packages = mkMerge [
      # ════════════════════════════════════════════════════════════════════════
      # Core Tools
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.core.enable (with pkgs; [
        pandoc typst gnuplot graphviz jq yq csvkit miller
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # LaTeX
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.latex.enable (
        let
          scheme = {
            minimal = pkgs.texlive.combined.scheme-minimal;
            small = pkgs.texlive.combined.scheme-small;
            medium = pkgs.texlive.combined.scheme-medium;
            full = pkgs.texlive.combined.scheme-full;
          }.${cfg.latex.scheme};
        in [ scheme pkgs.texmaker ]
      ))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Core (always with biology.enable)
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.biology.enable (with pkgs; [
        # Universal bio CLI tools
        seqkit          # FASTA/Q toolkit
        seqtk           # Sequence processing
        emboss          # EMBOSS suite
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Genomics (NGS pipeline)
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.genomics.enable) (with pkgs; [
        # Alignment
        blast           # NCBI BLAST+
        bwa             # Short-read alignment
        bowtie2         # Short-read alignment
        minimap2        # Long-read alignment
        diamond         # Fast protein alignment
        
        # SAM/BAM/VCF processing
        samtools
        bcftools
        bedtools
        htslib
        
        # Quality control
        fastqc
        multiqc
        fastp
        trimmomatic
        cutadapt
        
        # Assembly
        spades
        
        # Variant calling
        freebayes
        # gatk          # May need unfree
        
        # Annotation
        snpeff
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Transcriptomics (RNA-seq)
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.transcriptomics.enable) (with pkgs; [
        # Alignment
        star            # RNA-seq aligner
        hisat2          # Splice-aware aligner
        
        # Quantification
        salmon          # Fast transcript quant
        kallisto        # Pseudoalignment
        rsem
        subread         # featureCounts
        htseq
        
        # QC
        rseqc           # RNA-seq QC
        qualimap        # BAM QC
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Single-Cell & Spatial
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.singleCell.enable) (with pkgs; [
        # Python single-cell (from overlay)
        python-singlecell
        
        # Cell Ranger (if available)
        # cellranger    # 10x Genomics - typically manual install
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - CRISPR & Gene Editing
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.crispr.enable) (with pkgs; [
        # Python CRISPR stack (from overlay)
        python-crispr
        
        # Primer/oligo design
        primer3
        
        # Off-target analysis (alignment)
        blast
        bwa
        bowtie2
        
        # Genome browser
        igv
        
        # Data processing
        samtools
        bcftools
        bedtools
        
        # R CRISPR packages (from overlay)
        r-crispr
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Structural Biology
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.structural.enable) (with pkgs; [
        # Structure visualization
        pymol
        chimera
        vmd
        
        # Python structural stack (from overlay)
        python-structural
        
        # RNA structure
        viennarna
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Phylogenetics
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.phylogenetics.enable) (with pkgs; [
        # Tree building
        raxml
        iqtree
        mrbayes
        fasttree
        
        # MSA
        mafft
        muscle
        clustal-omega
        
        # Tree visualization
        figtree
        
        # Utilities
        trimal
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Metagenomics
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.metagenomics.enable) (with pkgs; [
        # Taxonomic profiling
        kraken2
        bracken
        
        # Assembly
        megahit
        
        # Binning
        metabat2
        
        # Annotation
        prokka
        prodigal
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Synthetic Biology
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.synbio.enable) (with pkgs; [
        # Python sysbio stack (from overlay)
        python-sysbio
        
        # Sequence tools
        emboss
        seqkit
        primer3
        
        # RNA structure
        viennarna
        
        # Visualization
        graphviz
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Molecular Dynamics
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.molecularDynamics.enable) (with pkgs; [
        (cudaPkg gromacs gromacs-cuda-full)
        vmd
        pymol
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - GUI Applications
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.gui.enable) (with pkgs; [
        # Genome browser
        igv
        
        # Structure visualization
        pymol
        
        # Image analysis
        fiji
        
        # Sequence/alignment viewer
        jalview
        ugene
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # BIOLOGY - Bioconductor (R)
      # ════════════════════════════════════════════════════════════════════════
      (mkIf (cfg.biology.enable && cfg.biology.bioconductor.enable) [
        pkgs.r-bioconductor
      ])

      # ════════════════════════════════════════════════════════════════════════
      # Chemistry
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.chemistry.enable (with pkgs; [
        avogadro2 openbabel jmol
        (cudaPkg gromacs gromacsCuda)
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # Physics
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.physics.enable (with pkgs; [
        stellarium kstars
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # Geospatial
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.geospatial.enable (with pkgs; [
        qgis grass gdal proj geos
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # Mathematics
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.mathematics.enable (with pkgs; [
        maxima wxmaxima octave
      ]))

      # ════════════════════════════════════════════════════════════════════════
      # Electronics
      # ════════════════════════════════════════════════════════════════════════
      (mkIf cfg.electronics.enable (with pkgs; [
        kicad ngspice gtkwave
      ]))

      cfg.extraPackages
    ];

    # ══════════════════════════════════════════════════════════════════════════
    # Environment Variables
    # ══════════════════════════════════════════════════════════════════════════
    home.sessionVariables = mkMerge [
      (mkIf cfg.geospatial.enable {
        GDAL_DATA = "${pkgs.gdal}/share/gdal";
        PROJ_DATA = "${pkgs.proj}/share/proj";
      })
      (mkIf cfg.chemistry.enable {
        BABEL_DATADIR = "${pkgs.openbabel}/share/openbabel";
      })
      (mkIf (cfg.biology.enable && cfg.biology.genomics.enable) {
        # Common bioinformatics paths
        BLASTDB = "$HOME/.local/share/blast/db";
      })
    ];

    # ══════════════════════════════════════════════════════════════════════════
    # Shell Aliases
    # ══════════════════════════════════════════════════════════════════════════
    programs.bash.shellAliases = mkMerge [
      (mkIf cfg.core.enable {
        md2pdf = "pandoc -o output.pdf";
        csv-view = "mlr --icsv --opprint cat";
      })
      (mkIf cfg.biology.enable {
        # FASTA/Q quick stats
        fq-stats = "seqkit stats";
        fq-head = "seqkit head -n 10";
        fa-grep = "seqkit grep -r -p";
        
        # SAM/BAM
        bam-stats = "samtools flagstat";
        bam-view = "samtools view -h";
        
        # VCF
        vcf-stats = "bcftools stats";
        vcf-view = "bcftools view -h";
      })
    ];

    programs.zsh.shellAliases = mkMerge [
      (mkIf cfg.core.enable {
        md2pdf = "pandoc -o output.pdf";
        csv-view = "mlr --icsv --opprint cat";
      })
      (mkIf cfg.biology.enable {
        fq-stats = "seqkit stats";
        fq-head = "seqkit head -n 10";
        fa-grep = "seqkit grep -r -p";
        bam-stats = "samtools flagstat";
        bam-view = "samtools view -h";
        vcf-stats = "bcftools stats";
        vcf-view = "bcftools view -h";
      })
    ];
  };
}
