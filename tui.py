#!/usr/bin/env python3
"""
Science Flake TUI - Interactive Module Selector and Guide
Version: 1.0.0

Usage:
    python tui.py              Interactive selector
    python tui.py --exec       Output shell command (for eval)
    python tui.py --list       List all modules
    python tui.py --preset bio Full biology stack
    python tui.py --help       Show help

Requires: Python 3.8+ (curses included in stdlib)
"""

import curses
import sys
import os
import signal
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum, auto
from collections import OrderedDict

VERSION = "1.0.0"
MIN_WIDTH = 60
MIN_HEIGHT = 20

# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

class Category(Enum):
    CORE = ("Core", 1)
    BIOLOGY = ("Biology", 2)
    AIML = ("AI/ML", 3)
    SCIENCE = ("Science", 4)
    
    def __init__(self, display_name: str, order: int):
        self.display_name = display_name
        self.order = order

@dataclass
class Module:
    id: str
    name: str
    category: Category
    description: str
    packages: List[str]
    quick_start: List[str]
    pip_extras: List[str] = field(default_factory=list)
    r_packages: List[str] = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE REGISTRY (ordered by category)
# ══════════════════════════════════════════════════════════════════════════════

MODULES: Dict[str, Module] = OrderedDict([
    # ── CORE ──────────────────────────────────────────────────────────────────
    ("core", Module(
        id="core",
        name="Core Tools",
        category=Category.CORE,
        description="Essential data processing, documentation, and Python scientific stack",
        packages=["jq", "yq", "csvkit", "miller", "pandoc", "typst", "gnuplot", 
                  "graphviz", "numpy", "scipy", "pandas", "matplotlib", "jupyter"],
        quick_start=[
            "mlr --icsv --opprint cat data.csv   # Pretty-print CSV",
            "jq '.[] | .name' data.json          # Parse JSON",
            "pandoc doc.md -o doc.pdf            # Convert documents",
            "jupyter lab                         # Start notebooks",
        ],
    )),
    
    # ── BIOLOGY ───────────────────────────────────────────────────────────────
    ("bio", Module(
        id="bio",
        name="Bioinformatics",
        category=Category.BIOLOGY,
        description="Core genomics: alignment, variant calling, NGS processing",
        packages=["blast", "bwa", "bowtie2", "minimap2", "diamond", "samtools", 
                  "bcftools", "bedtools", "fastqc", "multiqc", "fastp", "igv"],
        quick_start=[
            "seqkit stats *.fastq.gz             # FASTQ statistics",
            "bwa mem ref.fa R1.fq R2.fq | samtools sort -o out.bam",
            "samtools flagstat aligned.bam       # Alignment stats",
            "bcftools stats variants.vcf         # Variant stats",
        ],
    )),
    ("crispr", Module(
        id="crispr",
        name="CRISPR/Gene Editing",
        category=Category.BIOLOGY,
        description="Guide RNA design, off-target analysis, editing quantification",
        packages=["primer3", "blast", "bwa", "bowtie2", "samtools", "bcftools", 
                  "bedtools", "igv", "biopython", "primer3-py"],
        quick_start=[
            "seqkit locate -p '.GG' target.fa    # Find PAM (SpCas9)",
            "blastn -query guides.fa -db genome -outfmt 6",
            "primer3_core < design.txt           # Design primers",
        ],
        pip_extras=["crispresso2", "chopchop", "cas-offinder"],
    )),
    ("singlecell", Module(
        id="singlecell",
        name="Single-Cell & Spatial",
        category=Category.BIOLOGY,
        description="scRNA-seq analysis, trajectory inference, spatial transcriptomics",
        packages=["scanpy", "anndata", "muon", "scvelo", "cellrank", "squidpy",
                  "harmonypy", "bbknn", "leidenalg"],
        quick_start=[
            "import scanpy as sc",
            "adata = sc.read_10x_mtx('matrix/')",
            "sc.pp.filter_cells(adata, min_genes=200)",
            "sc.tl.pca(adata); sc.pp.neighbors(adata); sc.tl.umap(adata)",
        ],
        pip_extras=["scvi-tools", "celltypist", "cellxgene"],
    )),
    ("rnaseq", Module(
        id="rnaseq",
        name="RNA-Seq",
        category=Category.BIOLOGY,
        description="Bulk transcriptomics: alignment, quantification, differential expression",
        packages=["star", "hisat2", "salmon", "kallisto", "rsem", "subread", "htseq", "fastqc"],
        quick_start=[
            "salmon quant -i idx -l A -1 R1.fq.gz -2 R2.fq.gz -o quant",
            "STAR --genomeDir idx --readFilesIn R1.fq --outSAMtype BAM SortedByCoordinate",
            "# R: dds <- DESeqDataSetFromTximport(txi, colData, ~condition)",
        ],
        r_packages=["DESeq2", "edgeR", "limma", "tximport"],
    )),
    ("protein", Module(
        id="protein",
        name="Structural Biology",
        category=Category.BIOLOGY,
        description="Protein structure visualization, molecular dynamics analysis",
        packages=["pymol", "chimera", "vmd", "viennarna", "biopython", "mdanalysis"],
        quick_start=[
            "pymol structure.pdb                 # Visualize structure",
            "echo 'GCGCUUCGAGCG' | RNAfold       # RNA secondary structure",
            "# Python: from Bio.PDB import PDBParser",
        ],
        pip_extras=["esm", "openfold", "colabfold"],
    )),
    ("synbio", Module(
        id="synbio",
        name="Synthetic Biology",
        category=Category.BIOLOGY,
        description="Metabolic engineering, pathway design, constraint-based modeling",
        packages=["emboss", "seqkit", "primer3", "viennarna", "cobra", "tellurium"],
        quick_start=[
            "import cobra",
            "model = cobra.io.read_sbml_model('model.xml')",
            "model.optimize()                    # FBA optimization",
            "seqkit seq -rp sequence.fa          # Reverse complement",
        ],
        pip_extras=["cameo", "optlang"],
    )),
    
    # ── AI/ML ─────────────────────────────────────────────────────────────────
    ("ml", Module(
        id="ml",
        name="Machine Learning",
        category=Category.AIML,
        description="Deep learning, transformers, LLM fine-tuning",
        packages=["torch", "torchvision", "torchaudio", "transformers", "datasets",
                  "scikit-learn", "xgboost", "tensorboard", "wandb", "llama-cpp"],
        quick_start=[
            "from transformers import AutoModelForCausalLM",
            "model = AutoModelForCausalLM.from_pretrained('...')",
            "llama-cli -m model.gguf -p 'Hello'  # Local inference",
        ],
        pip_extras=["peft", "trl", "bitsandbytes", "flash-attn", "vllm"],
    )),
    ("rl", Module(
        id="rl",
        name="Reinforcement Learning",
        category=Category.AIML,
        description="RL algorithms, environments, multi-agent systems",
        packages=["gymnasium", "stable-baselines3", "pettingzoo", "torch",
                  "tensorboard", "wandb", "hydra-core"],
        quick_start=[
            "from stable_baselines3 import PPO",
            "model = PPO('MlpPolicy', 'CartPole-v1')",
            "model.learn(total_timesteps=10000)",
            "model.save('ppo_cartpole')",
        ],
        pip_extras=["pufferlib", "mujoco", "ray[rllib]"],
    )),
    
    # ── SCIENCE ───────────────────────────────────────────────────────────────
    ("geo", Module(
        id="geo",
        name="Geospatial",
        category=Category.SCIENCE,
        description="GIS analysis, remote sensing, spatial data processing",
        packages=["qgis", "grass", "gdal", "proj", "geos", "geopandas", "rasterio"],
        quick_start=[
            "gdal_translate -of GTiff input.img output.tif",
            "ogr2ogr -f GeoJSON out.json input.shp",
            "import geopandas as gpd; gdf = gpd.read_file('data.shp')",
        ],
        pip_extras=["leafmap", "geemap"],
    )),
    ("chem", Module(
        id="chem",
        name="Chemistry",
        category=Category.SCIENCE,
        description="Molecular visualization, dynamics, cheminformatics",
        packages=["avogadro2", "openbabel", "jmol", "gromacs", "rdkit"],
        quick_start=[
            "obabel molecule.sdf -O molecule.pdb # Format conversion",
            "obabel -:'CCO' -O ethanol.pdb --gen3d",
            "gmx pdb2gmx -f protein.pdb -o out.gro",
        ],
        pip_extras=["deepchem"],
    )),
])

# ══════════════════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, Tuple[str, Set[str]]] = {
    "bio": ("Full Biology Stack", {"core", "bio", "crispr", "singlecell", "rnaseq", "protein", "synbio"}),
    "ml": ("Full ML Stack", {"core", "ml", "rl"}),
    "genomics": ("Genomics Pipeline", {"core", "bio", "rnaseq"}),
    "crispr-full": ("CRISPR Workflow", {"core", "bio", "crispr"}),
    "data": ("Data Science", {"core", "ml", "geo"}),
    "minimal": ("Minimal (Core Only)", {"core"}),
}

# ══════════════════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = 1
    SELECTED = 2
    ACTIVE = 3
    CATEGORY = 4
    HELP = 5
    SUCCESS = 6
    WARNING = 7
    DIM = 8
    BORDER = 9
    TITLE = 10
    ERROR = 11
    SCROLLBAR = 12

def init_colors():
    """Initialize color pairs with fallback"""
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    
    pairs = [
        (Colors.HEADER, curses.COLOR_CYAN, bg),
        (Colors.SELECTED, curses.COLOR_BLACK, curses.COLOR_CYAN),
        (Colors.ACTIVE, curses.COLOR_GREEN, bg),
        (Colors.CATEGORY, curses.COLOR_YELLOW, bg),
        (Colors.HELP, curses.COLOR_WHITE, bg),
        (Colors.SUCCESS, curses.COLOR_GREEN, bg),
        (Colors.WARNING, curses.COLOR_YELLOW, bg),
        (Colors.DIM, curses.COLOR_WHITE, bg),
        (Colors.BORDER, curses.COLOR_BLUE, bg),
        (Colors.TITLE, curses.COLOR_MAGENTA, bg),
        (Colors.ERROR, curses.COLOR_RED, bg),
        (Colors.SCROLLBAR, curses.COLOR_BLUE, bg),
    ]
    
    for pair_id, fg, bg_color in pairs:
        try:
            curses.init_pair(pair_id, fg, bg_color)
        except curses.error:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# TUI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class ScienceTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.selected: Set[str] = {"core"}
        self.cursor = 0
        self.module_list = list(MODULES.keys())
        self.scroll_offset = 0
        self.view = "main"
        self.message = ""
        self.message_type = "info"
        self.message_timer = 0
        self.search_query = ""
        self.search_mode = False
        
        curses.curs_set(0)
        init_colors()
        self.stdscr.timeout(100)
        signal.signal(signal.SIGWINCH, lambda s, f: None)
        
    def run(self) -> Optional[Set[str]]:
        while True:
            h, w = self.stdscr.getmaxyx()
            
            if h < MIN_HEIGHT or w < MIN_WIDTH:
                self._draw_size_error(h, w)
                key = self.stdscr.getch()
                if key in (ord('q'), ord('Q')):
                    return None
                continue
            
            self.draw()
            
            if self.message_timer > 0:
                self.message_timer -= 1
                if self.message_timer == 0:
                    self.message = ""
            
            key = self.stdscr.getch()
            if key == -1:
                continue
            
            action = self.handle_input(key)
            if action == "quit":
                return None
            elif action == "launch":
                return self.selected
    
    def _draw_size_error(self, h: int, w: int):
        self.stdscr.clear()
        msg = f"Terminal too small ({w}x{h})"
        req = f"Minimum: {MIN_WIDTH}x{MIN_HEIGHT}"
        try:
            self.stdscr.attron(curses.color_pair(Colors.ERROR))
            self.stdscr.addstr(h//2, max(0, (w-len(msg))//2), msg[:w-1])
            self.stdscr.addstr(h//2+1, max(0, (w-len(req))//2), req[:w-1])
            self.stdscr.attroff(curses.color_pair(Colors.ERROR))
        except curses.error:
            pass
        self.stdscr.refresh()
    
    def handle_input(self, key) -> Optional[str]:
        if self.search_mode:
            return self._handle_search_input(key)
        handlers = {
            "main": self._handle_main_input,
            "detail": self._handle_detail_input,
            "presets": self._handle_presets_input,
            "confirm": self._handle_confirm_input,
        }
        return handlers.get(self.view, lambda k: None)(key)
    
    def _handle_search_input(self, key) -> Optional[str]:
        if key == 27:
            self.search_mode = False
            self.search_query = ""
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self.search_query = self.search_query[:-1]
        elif key in (ord('\n'), 10, 13):
            self.search_mode = False
            self._apply_search()
        elif 32 <= key <= 126:
            self.search_query += chr(key)
        return None
    
    def _apply_search(self):
        query = self.search_query.lower()
        for i, mod_id in enumerate(self.module_list):
            mod = MODULES[mod_id]
            if query in mod_id.lower() or query in mod.name.lower():
                self.cursor = i
                self._adjust_scroll()
                break
        self.search_query = ""
    
    def _handle_main_input(self, key) -> Optional[str]:
        if key in (ord('q'), ord('Q')):
            return "quit"
        elif key in (curses.KEY_UP, ord('k')):
            self.cursor = max(0, self.cursor - 1)
            self._adjust_scroll()
        elif key in (curses.KEY_DOWN, ord('j')):
            self.cursor = min(len(self.module_list) - 1, self.cursor + 1)
            self._adjust_scroll()
        elif key in (ord(' '), ord('\n'), curses.KEY_ENTER, 10, 13):
            self._toggle_module()
        elif key in (ord('d'), ord('D'), curses.KEY_RIGHT, ord('l')):
            self.view = "detail"
        elif key in (ord('r'), ord('R')):
            self.view = "confirm"
        elif key in (ord('a'), ord('A')):
            self._select_all()
        elif key in (ord('n'), ord('N')):
            self._select_none()
        elif key in (ord('p'), ord('P')):
            self.view = "presets"
        elif key == ord('/'):
            self.search_mode = True
            self.search_query = ""
        elif key in (ord('?'), ord('h'), ord('H')):
            self.view = "detail"
        elif key == curses.KEY_PPAGE:
            self.cursor = max(0, self.cursor - 10)
            self._adjust_scroll()
        elif key == curses.KEY_NPAGE:
            self.cursor = min(len(self.module_list) - 1, self.cursor + 10)
            self._adjust_scroll()
        return None
    
    def _handle_detail_input(self, key) -> Optional[str]:
        if key in (ord('q'), 27, curses.KEY_LEFT, ord('h')):
            self.view = "main"
        elif key in (curses.KEY_UP, ord('k')):
            self.cursor = max(0, self.cursor - 1)
            self._adjust_scroll()
        elif key in (curses.KEY_DOWN, ord('j')):
            self.cursor = min(len(self.module_list) - 1, self.cursor + 1)
            self._adjust_scroll()
        elif key == ord(' '):
            self._toggle_module()
        return None
    
    def _handle_presets_input(self, key) -> Optional[str]:
        if key in (ord('q'), 27, curses.KEY_LEFT):
            self.view = "main"
        elif ord('1') <= key <= ord('6'):
            preset_idx = key - ord('1')
            preset_keys = list(PRESETS.keys())
            if preset_idx < len(preset_keys):
                preset_name = preset_keys[preset_idx]
                _, modules = PRESETS[preset_name]
                self.selected = modules.copy()
                self._set_message(f"Applied: {preset_name}", "success")
                self.view = "main"
        return None
    
    def _handle_confirm_input(self, key) -> Optional[str]:
        if key in (ord('y'), ord('Y'), ord('\n'), 10, 13):
            return "launch"
        elif key in (ord('n'), ord('N'), 27, ord('q')):
            self.view = "main"
        return None
    
    def _toggle_module(self):
        mod_id = self.module_list[self.cursor]
        if mod_id == "core":
            self._set_message("Core is always included", "warning")
            return
        if mod_id in self.selected:
            self.selected.remove(mod_id)
            self._set_message(f"Removed: {MODULES[mod_id].name}", "info")
        else:
            self.selected.add(mod_id)
            self._set_message(f"Added: {MODULES[mod_id].name}", "success")
    
    def _select_all(self):
        self.selected = set(MODULES.keys())
        self._set_message("Selected all modules", "success")
    
    def _select_none(self):
        self.selected = {"core"}
        self._set_message("Reset to core only", "warning")
    
    def _set_message(self, msg: str, msg_type: str = "info"):
        self.message = msg
        self.message_type = msg_type
        self.message_timer = 30
    
    def _adjust_scroll(self):
        h, _ = self.stdscr.getmaxyx()
        visible = h - 14
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + visible:
            self.scroll_offset = self.cursor - visible + 1
    
    def draw(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        views = {
            "main": self._draw_main,
            "detail": self._draw_detail,
            "presets": self._draw_presets,
            "confirm": self._draw_confirm,
        }
        views.get(self.view, self._draw_main)(h, w)
        self.stdscr.refresh()
    
    def _safe_addstr(self, y: int, x: int, text: str, attr: int = 0):
        h, w = self.stdscr.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            try:
                self.stdscr.addstr(y, x, text[:max(0, w-x-1)], attr)
            except curses.error:
                pass
    
    def _draw_main(self, h: int, w: int):
        self._draw_header(w)
        self._draw_module_list(h, w, 4)
        self._draw_scrollbar(h, w)
        self._draw_status_bar(h, w)
        self._draw_footer(h, w)
    
    def _draw_header(self, w: int):
        title = "SCIENCE FLAKE"
        self._safe_addstr(0, 2, "┏━━━ ", curses.color_pair(Colors.BORDER) | curses.A_BOLD)
        self._safe_addstr(0, 7, title, curses.color_pair(Colors.HEADER) | curses.A_BOLD)
        self._safe_addstr(0, 7 + len(title), " ━━━┓", curses.color_pair(Colors.BORDER) | curses.A_BOLD)
        self._safe_addstr(0, w - 8, f"v{VERSION}", curses.color_pair(Colors.DIM))
        
        if self.search_mode:
            self._safe_addstr(2, 2, f"/{self.search_query}_", curses.color_pair(Colors.WARNING))
        else:
            self._safe_addstr(2, 2, "Module Selector & Guide", curses.color_pair(Colors.DIM))
    
    def _draw_module_list(self, h: int, w: int, start_y: int):
        visible = h - 14
        current_cat = None
        y = start_y
        list_idx = 0
        
        for idx, mod_id in enumerate(self.module_list):
            module = MODULES[mod_id]
            
            if module.category != current_cat:
                current_cat = module.category
                if list_idx >= self.scroll_offset and y < start_y + visible:
                    self._safe_addstr(y, 2, f"▸ {current_cat.display_name}", 
                                     curses.color_pair(Colors.CATEGORY) | curses.A_BOLD)
                    y += 1
                list_idx += 1
            
            if list_idx < self.scroll_offset:
                list_idx += 1
                continue
            if y >= start_y + visible:
                break
            
            is_sel = mod_id in self.selected
            is_cur = idx == self.cursor
            
            chk = "[✓]" if is_sel else "[ ]"
            name = module.name[:18].ljust(18)
            desc = module.description[:w-30] if w > 30 else ""
            line = f"  {chk} {name} {desc}"
            
            if is_cur:
                attr = curses.color_pair(Colors.SELECTED)
                self._safe_addstr(y, 0, " " * (w-1), attr)
                self._safe_addstr(y, 0, line, attr)
            elif is_sel:
                self._safe_addstr(y, 0, line, curses.color_pair(Colors.ACTIVE))
            else:
                self._safe_addstr(y, 0, line, 0)
            
            y += 1
            list_idx += 1
    
    def _draw_scrollbar(self, h: int, w: int):
        list_h = h - 14
        total = len(self.module_list) + 4
        if total <= list_h:
            return
        
        bar_h = max(1, int(list_h * list_h / total))
        bar_pos = int(self.scroll_offset * (list_h - bar_h) / max(1, total - list_h))
        
        x = w - 2
        for i in range(list_h):
            char = "█" if bar_pos <= i < bar_pos + bar_h else "│"
            color = Colors.SCROLLBAR if char == "█" else Colors.DIM
            self._safe_addstr(4 + i, x, char, curses.color_pair(color))
    
    def _draw_status_bar(self, h: int, w: int):
        y = h - 5
        self._safe_addstr(y, 0, "─" * (w-1), curses.color_pair(Colors.BORDER))
        
        y += 1
        names = ", ".join(sorted(self.selected))
        if len(names) > w - 18:
            names = names[:w-21] + "..."
        self._safe_addstr(y, 2, f"Selected ({len(self.selected)}): {names}", curses.color_pair(Colors.SUCCESS))
        
        if self.message:
            y += 1
            color = {"success": Colors.SUCCESS, "warning": Colors.WARNING, "error": Colors.ERROR}.get(self.message_type, Colors.DIM)
            self._safe_addstr(y, 2, self.message, curses.color_pair(color))
    
    def _draw_footer(self, h: int, w: int):
        shortcuts = [("↑↓", "nav"), ("Space", "sel"), ("d", "detail"), ("p", "preset"), ("/", "find"), ("r", "run"), ("q", "quit")]
        x = 2
        for key, desc in shortcuts:
            if x + len(key) + len(desc) + 3 > w - 2:
                break
            self._safe_addstr(h - 2, x, key, curses.color_pair(Colors.HEADER) | curses.A_BOLD)
            self._safe_addstr(h - 2, x + len(key), f":{desc}", curses.color_pair(Colors.DIM))
            x += len(key) + len(desc) + 2
    
    def _draw_detail(self, h: int, w: int):
        mod = MODULES[self.module_list[self.cursor]]
        self._draw_header(w)
        
        y = 4
        is_sel = mod.id in self.selected
        status = "✓ SELECTED" if is_sel else "○ NOT SELECTED"
        self._safe_addstr(y, 2, f"▶ {mod.name}", curses.color_pair(Colors.TITLE) | curses.A_BOLD)
        self._safe_addstr(y, 5 + len(mod.name), f"[{status}]", curses.color_pair(Colors.SUCCESS if is_sel else Colors.DIM))
        y += 2
        
        self._safe_addstr(y, 2, mod.description[:w-4], 0)
        y += 2
        
        self._safe_addstr(y, 2, "Packages:", curses.color_pair(Colors.CATEGORY) | curses.A_BOLD)
        y += 1
        for line in self._wrap(", ".join(mod.packages), w - 6):
            if y >= h - 10:
                break
            self._safe_addstr(y, 4, line, curses.color_pair(Colors.DIM))
            y += 1
        y += 1
        
        if mod.r_packages:
            self._safe_addstr(y, 2, "R:", curses.color_pair(Colors.CATEGORY) | curses.A_BOLD)
            self._safe_addstr(y, 5, ", ".join(mod.r_packages), curses.color_pair(Colors.DIM))
            y += 2
        
        self._safe_addstr(y, 2, "Quick Start:", curses.color_pair(Colors.CATEGORY) | curses.A_BOLD)
        y += 1
        for cmd in mod.quick_start:
            if y >= h - 6:
                break
            self._safe_addstr(y, 4, cmd[:w-6], curses.color_pair(Colors.SUCCESS))
            y += 1
        y += 1
        
        if mod.pip_extras:
            self._safe_addstr(y, 2, "pip extras:", curses.color_pair(Colors.WARNING) | curses.A_BOLD)
            y += 1
            self._safe_addstr(y, 4, f"pip install {' '.join(mod.pip_extras)}"[:w-6], curses.color_pair(Colors.WARNING))
        
        self._safe_addstr(h - 2, 2, "←/q:back  ↑↓:nav  Space:toggle", curses.color_pair(Colors.DIM))
    
    def _draw_presets(self, h: int, w: int):
        self._draw_header(w)
        y = 4
        self._safe_addstr(y, 2, "Quick Presets", curses.color_pair(Colors.TITLE) | curses.A_BOLD)
        y += 2
        
        for i, (pid, (name, mods)) in enumerate(PRESETS.items()):
            mod_list = ", ".join(sorted(mods))
            if len(mod_list) > w - 30:
                mod_list = mod_list[:w-33] + "..."
            self._safe_addstr(y, 2, f"[{i+1}]", curses.color_pair(Colors.HEADER) | curses.A_BOLD)
            self._safe_addstr(y, 6, name, curses.color_pair(Colors.SUCCESS))
            y += 1
            self._safe_addstr(y, 6, mod_list, curses.color_pair(Colors.DIM))
            y += 2
        
        self._safe_addstr(h - 2, 2, "1-6:select  q/←:back", curses.color_pair(Colors.DIM))
    
    def _draw_confirm(self, h: int, w: int):
        self._draw_header(w)
        box_w, box_h = min(50, w - 4), 8
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2
        
        self._safe_addstr(box_y, box_x, "┌" + "─" * (box_w - 2) + "┐", curses.color_pair(Colors.BORDER))
        for i in range(1, box_h - 1):
            self._safe_addstr(box_y + i, box_x, "│" + " " * (box_w - 2) + "│", curses.color_pair(Colors.BORDER))
        self._safe_addstr(box_y + box_h - 1, box_x, "└" + "─" * (box_w - 2) + "┘", curses.color_pair(Colors.BORDER))
        
        self._safe_addstr(box_y + 2, box_x + (box_w - 18) // 2, "Launch Environment?", curses.color_pair(Colors.TITLE) | curses.A_BOLD)
        self._safe_addstr(box_y + 4, box_x + (box_w - 20) // 2, f"{len(self.selected)} modules selected", curses.color_pair(Colors.DIM))
        self._safe_addstr(box_y + 6, box_x + (box_w - 11) // 2, "[Y]es  [N]o", curses.color_pair(Colors.SUCCESS))
    
    def _wrap(self, text: str, width: int) -> List[str]:
        words = text.split()
        lines, cur, cur_len = [], [], 0
        for w in words:
            if cur_len + len(w) + 1 <= width:
                cur.append(w)
                cur_len += len(w) + 1
            else:
                if cur:
                    lines.append(" ".join(cur))
                cur, cur_len = [w], len(w)
        if cur:
            lines.append(" ".join(cur))
        return lines

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def gen_cmd(selected: Set[str]) -> str:
    return f"nix-shell shell.nix --arg modules '[ {' '.join(f'\"{m}\"' for m in sorted(selected))} ]'"

def print_result(selected: Set[str]):
    print(f"\n{'═'*60}\n  SCIENCE FLAKE v{VERSION}\n{'═'*60}")
    print(f"\n  Modules ({len(selected)}): {', '.join(sorted(selected))}\n")
    print(f"  {gen_cmd(selected)}\n")
    print(f"{'─'*60}\n  Or: eval \"$(python tui.py --exec)\"\n{'═'*60}\n")

def main():
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print(f"""
Science Flake TUI v{VERSION}

Usage:
    python tui.py              Interactive selector
    python tui.py --exec       Output command for eval
    python tui.py --list       List modules
    python tui.py --preset X   Apply preset

Keys: ↑↓/jk:nav  Space:toggle  d:detail  p:presets  /:search  r:run  q:quit

Presets: {', '.join(PRESETS.keys())}
""")
        return
    
    if "--list" in args:
        cat = None
        for m in MODULES.values():
            if m.category != cat:
                cat = m.category
                print(f"\n{cat.display_name}:")
            print(f"  {'✓' if m.id == 'core' else ' '} {m.id:12} {m.name}")
        return
    
    if "--preset" in args:
        try:
            name = args[args.index("--preset") + 1]
            if name in PRESETS:
                _, mods = PRESETS[name]
                print(gen_cmd(mods)) if "--exec" in args else print_result(mods)
            else:
                print(f"Unknown preset. Available: {', '.join(PRESETS.keys())}")
                sys.exit(1)
        except IndexError:
            print("--preset requires name")
            sys.exit(1)
        return
    
    if "--exec" in args:
        try:
            sel = curses.wrapper(lambda s: ScienceTUI(s).run())
            if sel:
                print(gen_cmd(sel))
        except KeyboardInterrupt:
            pass
        return
    
    try:
        sel = curses.wrapper(lambda s: ScienceTUI(s).run())
        if sel:
            print_result(sel)
    except KeyboardInterrupt:
        print("\nCancelled.")

if __name__ == "__main__":
    main()
