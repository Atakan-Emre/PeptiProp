"""Unified pipeline for protein-peptide interaction analysis"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import shutil

import numpy as np

from ..data.models import Complex, StructureSource, StructureOrigin
from ..data.fetchers.rcsb_fetcher import RCSBFetcher
from ..structure.parsers.mmcif_parser import StructureParser
from ..interaction import (
    ArpeggioWrapper,
    PLIPWrapper,
    InteractionMerger,
    ContactMatrixGenerator,
    InteractionFingerprintBuilder,
    InteractionSet,
    InteractionType,
    StandardizedInteraction,
)
from ..visualization import (
    ContactMapPlotter,
    PyMOLRenderer,
    Peptide2DRenderer,
    Viewer3DMol,
    ReportBuilder
)


logger = logging.getLogger(__name__)


class PeptidQuantumPipeline:
    """
    Unified pipeline for protein-peptide interaction analysis
    
    Pipeline stages:
    1. Acquire structure (RCSB/PROPEDIA or local file)
    2. Parse and normalize (mmCIF parsing, chain classification)
    3. Extract pocket (residues within distance cutoff)
    4. Extract interactions (Arpeggio + PLIP)
    5. Build analysis (contact matrix, fingerprint)
    6. Render visuals (PyMOL, contact maps, 2D chemistry)
    7. Build report (HTML with 3Dmol.js viewer)
    """
    
    def __init__(
        self,
        output_base_dir: str | Path = "outputs",
        cache_dir: Optional[str | Path] = None
    ):
        """
        Initialize pipeline
        
        Args:
            output_base_dir: Base directory for outputs
            cache_dir: Cache directory for downloaded structures
        """
        self.output_base_dir = Path(output_base_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        
        # Initialize components
        self.rcsb_fetcher = RCSBFetcher(cache_dir=self.cache_dir / "rcsb")
        self.structure_parser = StructureParser()
        self.arpeggio = ArpeggioWrapper()
        self.plip = PLIPWrapper()
        self.merger = InteractionMerger()
        self.contact_generator = ContactMatrixGenerator()
        self.fingerprint_builder = InteractionFingerprintBuilder()
        self.contact_plotter = ContactMapPlotter()
        self.pymol_renderer = PyMOLRenderer()
        self.peptide_renderer = Peptide2DRenderer()
        self.viewer_builder = Viewer3DMol()
        self.report_builder = ReportBuilder()
        
        logger.info("PeptidQuantum pipeline initialized")
    
    def run(
        self,
        complex_id: Optional[str] = None,
        cif_path: Optional[str | Path] = None,
        protein_chain: Optional[str] = None,
        peptide_chain: Optional[str] = None,
        pocket_radius: float = 8.0,
        use_arpeggio: bool = True,
        use_plip: bool = True,
        generate_pymol: bool = True,
        generate_report: bool = True,
        generate_viewer: bool = True
    ) -> Dict:
        """
        Run complete pipeline
        
        Args:
            complex_id: PDB ID to fetch (if cif_path not provided)
            cif_path: Path to local CIF file
            protein_chain: Protein chain ID (auto-detect if None)
            peptide_chain: Peptide chain ID (auto-detect if None)
            pocket_radius: Pocket extraction radius in Angstroms
            use_arpeggio: Use Arpeggio for interaction extraction
            use_plip: Use PLIP for interaction extraction
            generate_pymol: Generate PyMOL figures
            generate_report: Generate HTML report
            generate_viewer: Generate standalone 3Dmol.js viewer
            
        Returns:
            Dictionary with pipeline results and output paths
        """
        logger.info("="*60)
        logger.info("Starting PeptidQuantum Pipeline")
        logger.info("="*60)
        
        # Stage 1: Acquire structure
        complex_obj = self._acquire_structure(complex_id, cif_path)
        if not complex_obj:
            logger.error("Failed to acquire structure")
            return {"status": "failed", "stage": "acquire_structure"}
        
        # Create output directory
        output_dir = self.output_base_dir / complex_obj.complex_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "structures").mkdir(exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
        (output_dir / "figures").mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        # Stage 2: Parse and normalize
        complex_obj = self._parse_and_normalize(complex_obj, protein_chain, peptide_chain)
        if not complex_obj:
            logger.error("Failed to parse structure")
            return {"status": "failed", "stage": "parse_structure"}
        
        # Copy structure file
        if complex_obj.structure_file:
            shutil.copy(
                complex_obj.structure_file,
                output_dir / "structures" / "complex.cif"
            )
        
        # Stage 3: Extract pocket
        pocket_complex = self._extract_pocket(complex_obj, pocket_radius)
        
        # Stage 4: Extract interactions
        interaction_set = self._extract_interactions(
            complex_obj,
            use_arpeggio=use_arpeggio,
            use_plip=use_plip,
            output_dir=output_dir
        )
        
        if not interaction_set or len(interaction_set.interactions) == 0:
            logger.warning("No interactions found")
        
        # Stage 5: Build analysis
        analysis_results = self._build_analysis(
            interaction_set,
            complex_obj,
            output_dir
        )
        
        # Stage 6: Render visuals
        visual_results = self._render_visuals(
            complex_obj,
            interaction_set,
            pocket_complex,
            output_dir,
            generate_pymol=generate_pymol
        )
        
        # Stage 7: Build report
        report_results = self._build_report(
            complex_obj,
            interaction_set,
            output_dir,
            generate_report=generate_report,
            generate_viewer=generate_viewer
        )
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        return {
            "status": "success",
            "complex_id": complex_obj.complex_id,
            "output_dir": str(output_dir),
            "num_interactions": len(interaction_set.interactions),
            "interaction_types": len(interaction_set.get_interaction_types()),
            "analysis": analysis_results,
            "visuals": visual_results,
            "reports": report_results
        }
    
    def _acquire_structure(
        self,
        complex_id: Optional[str],
        cif_path: Optional[str | Path]
    ) -> Optional[Complex]:
        """Stage 1: Acquire structure"""
        logger.info("Stage 1: Acquiring structure...")
        
        if cif_path:
            # Use local file
            cif_path = Path(cif_path)
            if not cif_path.exists():
                logger.error(f"CIF file not found: {cif_path}")
                return None
            
            logger.info(f"Using local file: {cif_path}")
            structure_file = cif_path
            complex_id = complex_id or cif_path.stem
            source = StructureSource.EXPERIMENTAL
            origin = StructureOrigin.RCSB
            
        elif complex_id:
            # Fetch from RCSB
            logger.info(f"Fetching {complex_id} from RCSB PDB...")
            structure_file = self.rcsb_fetcher.fetch_structure(complex_id, format="cif")
            
            if not structure_file:
                logger.error(f"Failed to fetch {complex_id}")
                return None
            
            source = StructureSource.EXPERIMENTAL
            origin = StructureOrigin.RCSB
            
        else:
            logger.error("Must provide either complex_id or cif_path")
            return None
        
        # Create complex object with metadata
        complex_obj = Complex(
            complex_id=complex_id,
            structure_source=source,
            structure_origin=origin,
            structure_file=str(structure_file)
        )
        
        logger.info(f"✓ Structure acquired: {complex_id}")
        return complex_obj
    
    def _parse_and_normalize(
        self,
        complex_obj: Complex,
        protein_chain: Optional[str],
        peptide_chain: Optional[str]
    ) -> Optional[Complex]:
        """Stage 2: Parse and normalize structure"""
        logger.info("Stage 2: Parsing and normalizing structure...")
        
        # Parse structure
        parsed_complex = self.structure_parser.parse_file(
            complex_obj.structure_file,
            complex_id=complex_obj.complex_id,
            structure_source=complex_obj.structure_source,
            structure_origin=complex_obj.structure_origin
        )
        
        if not parsed_complex:
            logger.error("Failed to parse structure")
            return None
        
        # Override chain classification if specified
        if protein_chain or peptide_chain:
            # Reclassify chains
            all_chains = parsed_complex.protein_chains + parsed_complex.peptide_chains
            parsed_complex.protein_chains = []
            parsed_complex.peptide_chains = []
            
            for chain in all_chains:
                if protein_chain and chain.chain_id == protein_chain:
                    chain.chain_type = "protein"
                    parsed_complex.protein_chains.append(chain)
                elif peptide_chain and chain.chain_id == peptide_chain:
                    chain.chain_type = "peptide"
                    parsed_complex.peptide_chains.append(chain)
                else:
                    # Keep original classification
                    if chain.chain_type == "protein":
                        parsed_complex.protein_chains.append(chain)
                    else:
                        parsed_complex.peptide_chains.append(chain)
        
        logger.info(f"✓ Parsed {len(parsed_complex.protein_chains)} protein chains, "
                   f"{len(parsed_complex.peptide_chains)} peptide chains")
        
        return parsed_complex
    
    def _extract_pocket(
        self,
        complex_obj: Complex,
        radius: float
    ) -> Optional[Complex]:
        """Stage 3: Extract binding pocket"""
        logger.info(f"Stage 3: Extracting pocket (radius={radius}Å)...")
        
        if not complex_obj.peptide_chains:
            logger.warning("No peptide chains found, skipping pocket extraction")
            return None
        
        # Extract pocket around first peptide chain
        peptide_chain = complex_obj.peptide_chains[0]
        pocket_complex = self.structure_parser.extract_pocket(
            complex_obj,
            peptide_chain.chain_id,
            radius=radius
        )
        
        n_pocket_residues = sum(len(c.residues) for c in pocket_complex.protein_chains)
        logger.info(f"✓ Extracted pocket: {n_pocket_residues} residues")
        
        return pocket_complex
    
    def _extract_interactions(
        self,
        complex_obj: Complex,
        use_arpeggio: bool,
        use_plip: bool,
        output_dir: Path
    ) -> InteractionSet:
        """Stage 4: Extract interactions"""
        logger.info("Stage 4: Extracting interactions...")
        
        interaction_sets = []
        
        # Arpeggio
        if use_arpeggio and self.arpeggio.is_available():
            logger.info("Running Arpeggio...")
            try:
                arpeggio_set = self.arpeggio.extract_interactions(
                    complex_obj,
                    output_dir=output_dir / "arpeggio_tmp"
                )
                interaction_sets.append(arpeggio_set)
                logger.info(f"  Arpeggio: {len(arpeggio_set.interactions)} interactions")
            except Exception as e:
                logger.warning(f"Arpeggio failed: {e}")
        elif use_arpeggio:
            logger.warning("Arpeggio not available, skipping")
        
        # PLIP
        if use_plip and self.plip.is_available():
            logger.info("Running PLIP...")
            try:
                plip_set = self.plip.extract_interactions(
                    complex_obj,
                    output_dir=output_dir / "plip_tmp"
                )
                interaction_sets.append(plip_set)
                logger.info(f"  PLIP: {len(plip_set.interactions)} interactions")
            except Exception as e:
                logger.warning(f"PLIP failed: {e}")
        elif use_plip:
            logger.warning("PLIP not available, skipping")
        
        # Merge interactions
        if interaction_sets:
            merged_set = self.merger.merge(*interaction_sets, strategy="union")
            logger.info(f"✓ Total interactions: {len(merged_set.interactions)}")
        else:
            logger.warning("No interaction extractors ran successfully, using geometric fallback")
            merged_set = self._build_geometric_fallback(
                complex_obj=complex_obj,
                distance_cutoff=8.0,
            )
            if len(merged_set.interactions) > 0:
                logger.info(f"✓ Geometric fallback interactions: {len(merged_set.interactions)}")
            else:
                logger.warning("Geometric fallback produced no contacts")
        
        return merged_set

    def _build_geometric_fallback(self, complex_obj: Complex, distance_cutoff: float = 8.0) -> InteractionSet:
        """
        Build a residue-contact interaction set from C-alpha distances.

        This keeps visualization/report generation useful even when Arpeggio/PLIP
        are unavailable on local machines.
        """
        if distance_cutoff <= 0:
            return InteractionSet(complex_id=complex_obj.complex_id, interactions=[])

        # Keep the minimum distance for each residue pair.
        pair_to_entry: Dict[Tuple[str, int, str, int], Dict[str, object]] = {}

        for protein_chain in complex_obj.protein_chains:
            if not protein_chain.residues:
                continue
            protein_coords = np.array([[r.x, r.y, r.z] for r in protein_chain.residues], dtype=np.float32)

            for peptide_chain in complex_obj.peptide_chains:
                if not peptide_chain.residues:
                    continue
                peptide_coords = np.array([[r.x, r.y, r.z] for r in peptide_chain.residues], dtype=np.float32)

                # Shape: [n_protein_res, n_peptide_res]
                dists = np.linalg.norm(
                    protein_coords[:, None, :] - peptide_coords[None, :, :],
                    axis=2,
                )
                protein_idx, peptide_idx = np.where(dists <= float(distance_cutoff))
                for p_i, pep_i in zip(protein_idx.tolist(), peptide_idx.tolist()):
                    p_res = protein_chain.residues[p_i]
                    pep_res = peptide_chain.residues[pep_i]
                    dist = float(dists[p_i, pep_i])
                    key = (
                        str(protein_chain.chain_id),
                        int(p_res.residue_number),
                        str(peptide_chain.chain_id),
                        int(pep_res.residue_number),
                    )
                    prev = pair_to_entry.get(key)
                    if prev is None or dist < float(prev["distance"]):
                        pair_to_entry[key] = {
                            "protein_chain": str(protein_chain.chain_id),
                            "protein_residue_id": int(p_res.residue_number),
                            "protein_residue_name": str(p_res.residue_name),
                            "peptide_chain": str(peptide_chain.chain_id),
                            "peptide_residue_id": int(pep_res.residue_number),
                            "peptide_residue_name": str(pep_res.residue_name),
                            "distance": dist,
                        }

        interactions: List[StandardizedInteraction] = []
        for entry in pair_to_entry.values():
            dist = float(entry["distance"])
            confidence = max(0.1, 1.0 - (dist / float(distance_cutoff)))
            interactions.append(
                StandardizedInteraction(
                    protein_chain=entry["protein_chain"],
                    protein_residue_id=entry["protein_residue_id"],
                    protein_residue_name=entry["protein_residue_name"],
                    peptide_chain=entry["peptide_chain"],
                    peptide_residue_id=entry["peptide_residue_id"],
                    peptide_residue_name=entry["peptide_residue_name"],
                    interaction_type=InteractionType.VDW,
                    distance=dist,
                    source_tool="geometric_fallback",
                    confidence=float(confidence),
                    raw_type="geometric_contact",
                )
            )

        return InteractionSet(complex_id=complex_obj.complex_id, interactions=interactions)
    
    def _build_analysis(
        self,
        interaction_set: InteractionSet,
        complex_obj: Complex,
        output_dir: Path
    ) -> Dict:
        """Stage 5: Build analysis"""
        logger.info("Stage 5: Building analysis...")
        
        results = {}
        
        if len(interaction_set.interactions) == 0:
            logger.warning("No interactions to analyze")
            return results
        
        # Get chains
        protein_chain = complex_obj.protein_chains[0].chain_id if complex_obj.protein_chains else None
        peptide_chain = complex_obj.peptide_chains[0].chain_id if complex_obj.peptide_chains else None
        
        if not (protein_chain and peptide_chain):
            logger.warning("Missing chain information")
            return results
        
        # Contact matrix
        logger.info("Generating contact matrix...")
        matrix, protein_res, peptide_res = self.contact_generator.generate_matrix(
            interaction_set,
            protein_chain,
            peptide_chain,
            aggregation="count"
        )
        
        if matrix.size > 0:
            self.contact_generator.save_csv(
                matrix,
                protein_res,
                peptide_res,
                output_dir / "data" / "residue_residue_matrix.csv"
            )
            results['contact_matrix'] = True
        
        # Interaction fingerprint
        logger.info("Building interaction fingerprint...")
        fingerprint = self.fingerprint_builder.build_fingerprint(interaction_set)
        self.fingerprint_builder.save_json(
            fingerprint,
            output_dir / "data" / "interaction_fingerprint.json"
        )
        results['fingerprint'] = True
        
        # Save contacts table
        logger.info("Saving contacts table...")
        interaction_set.save_tsv(output_dir / "data" / "contacts.tsv")
        results['contacts_table'] = True
        
        logger.info(f"✓ Analysis complete")
        return results
    
    def _render_visuals(
        self,
        complex_obj: Complex,
        interaction_set: InteractionSet,
        pocket_complex: Optional[Complex],
        output_dir: Path,
        generate_pymol: bool
    ) -> Dict:
        """Stage 6: Render visuals"""
        logger.info("Stage 6: Rendering visuals...")
        
        results = {}
        figures_dir = output_dir / "figures"
        
        # Get chains
        protein_chain = complex_obj.protein_chains[0].chain_id if complex_obj.protein_chains else None
        peptide_chain = complex_obj.peptide_chains[0].chain_id if complex_obj.peptide_chains else None
        
        # Contact maps
        if len(interaction_set.interactions) > 0 and protein_chain and peptide_chain:
            logger.info("Generating contact maps...")
            try:
                self.contact_plotter.plot_contact_map(
                    interaction_set,
                    protein_chain,
                    peptide_chain,
                    figures_dir / "contact_map.png"
                )
                results['contact_map'] = True
            except Exception as e:
                logger.warning(f"Contact map failed: {e}")
            
            try:
                self.contact_plotter.plot_contact_map_by_type(
                    interaction_set,
                    protein_chain,
                    peptide_chain,
                    figures_dir / "contact_map_by_type.png"
                )
                results['contact_map_by_type'] = True
            except Exception as e:
                logger.warning(f"Type-specific contact map failed: {e}")
            
            try:
                self.contact_plotter.plot_interaction_summary(
                    interaction_set,
                    figures_dir / "interaction_summary.png"
                )
                results['interaction_summary'] = True
            except Exception as e:
                logger.warning(f"Interaction summary failed: {e}")
        
        # PyMOL figures
        if generate_pymol and self.pymol_renderer.is_available():
            logger.info("Generating PyMOL figures...")
            
            try:
                self.pymol_renderer.render_overview(
                    complex_obj.structure_file,
                    peptide_chain,
                    figures_dir / "complex_overview.png"
                )
                results['pymol_overview'] = True
            except Exception as e:
                logger.warning(f"PyMOL overview failed: {e}")
            
            if pocket_complex and len(interaction_set.interactions) > 0:
                try:
                    pocket_residues = [
                        (c.chain_id, r.residue_number)
                        for c in pocket_complex.protein_chains
                        for r in c.residues
                    ]
                    self.pymol_renderer.render_pocket(
                        complex_obj.structure_file,
                        peptide_chain,
                        pocket_residues,
                        figures_dir / "pocket_zoom.png",
                        interactions=interaction_set
                    )
                    results['pymol_pocket'] = True
                except Exception as e:
                    logger.warning(f"PyMOL pocket failed: {e}")
            
            if len(interaction_set.interactions) > 0:
                try:
                    self.pymol_renderer.render_interactions(
                        complex_obj.structure_file,
                        interaction_set,
                        protein_chain,
                        peptide_chain,
                        figures_dir / "interaction_overlay.png"
                    )
                    results['pymol_interactions'] = True
                except Exception as e:
                    logger.warning(f"PyMOL interactions failed: {e}")
        elif generate_pymol:
            logger.warning("PyMOL not available, skipping PyMOL figures")
        
        # Peptide 2D
        if peptide_chain and complex_obj.peptide_chains:
            logger.info("Generating peptide 2D structure...")
            try:
                peptide_seq = complex_obj.peptide_chains[0].sequence
                protein_label = protein_chain if protein_chain else "N/A"
                self.peptide_renderer.from_sequence(
                    peptide_seq,
                    figures_dir / "peptide_2d.png",
                    title=(
                        f"{complex_obj.complex_id} | Protein {protein_label} "
                        f"| Peptide {peptide_chain}: {peptide_seq}"
                    )
                )
                results['peptide_2d'] = True
            except Exception as e:
                logger.warning(f"Peptide 2D rendering failed: {e}")
        
        logger.info(f"✓ Rendered {len(results)} visualizations")
        return results
    
    def _build_report(
        self,
        complex_obj: Complex,
        interaction_set: InteractionSet,
        output_dir: Path,
        generate_report: bool,
        generate_viewer: bool
    ) -> Dict:
        """Stage 7: Build report"""
        logger.info("Stage 7: Building report...")
        
        results = {}
        
        # Standalone viewer
        if generate_viewer:
            logger.info("Generating 3Dmol.js viewer...")
            try:
                self.viewer_builder.create_viewer(
                    complex_obj,
                    interaction_set,
                    output_html=output_dir / "viewer.html",
                    output_json=output_dir / "data" / "viewer_state.json"
                )
                results['viewer'] = True
            except Exception as e:
                logger.warning(f"Viewer generation failed: {e}")
        
        # HTML report
        if generate_report:
            logger.info("Generating HTML report...")
            try:
                self.report_builder.build(
                    complex_obj,
                    interaction_set,
                    assets_dir=output_dir / "figures",
                    output_html=output_dir / "report.html",
                    include_viewer=True
                )
                results['report'] = True
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")
        
        logger.info(f"✓ Report generation complete")
        return results
