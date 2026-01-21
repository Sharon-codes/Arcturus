import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class SpectrumObject:
    id: str
    source: str  # "PAHdb" | "PDS"
    x: np.ndarray
    y: np.ndarray
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UniversalParser:
    def __init__(self):
        self.supported_extensions = ['.csv', '.txt', '.dat', '.jdx', '.spc', '.tab', '.xml']

    def parse_pahdb_xml(self, filepath: str, limit: Optional[int] = None) -> List[SpectrumObject]:
        """Iteratively parses the theoretical PAHdb XML file to save memory."""
        spectra = []
        
        # Define the namespace
        ns = {'pahdb': 'http://www.astrochemistry.org/pahdb/theoretical'}
        
        try:
            print(f"Loading PAHdb: {filepath}", flush=True)
            pbar = tqdm(total=limit if limit else 100, desc="Parsing PAHdb", unit="species")
            
            context = ET.iterparse(filepath, events=('end',))
            for event, elem in context:
                # Match with namespace - check if tag ends with 'specie'
                if elem.tag.endswith('}specie') or elem.tag == 'specie':
                    uid = elem.get('uid')
                    
                    # Extract formula and weight (namespace-aware)
                    formula = None
                    weight = None
                    for child in elem:
                        tag = child.tag.split('}')[-1]  # Get tag without namespace
                        if tag == 'formula':
                            formula = child.text
                        elif tag == 'weight':
                            weight = child.text
                    
                    x_vals = []
                    y_vals = []
                    
                    # Find transitions element
                    transitions = None
                    for child in elem:
                        tag = child.tag.split('}')[-1]
                        if tag == 'transitions':  
                            transitions = child
                            break
                    
                    if transitions is not None:
                        # Extract mode data from transitions
                        for mode in transitions:
                            mode_tag = mode.tag.split('}')[-1]
                            if mode_tag == 'mode':
                                freq = None
                                intens = None
                                for prop in mode:
                                    prop_tag = prop.tag.split('}')[-1]
                                    if prop_tag == 'frequency':
                                        freq = prop.text
                                    elif prop_tag == 'intensity':
                                        intens = prop.text
                                
                                if freq and intens:
                                    try:
                                        x_vals.append(float(freq))
                                        y_vals.append(float(intens))
                                    except ValueError:
                                        pass  # Skip invalid values

                    if x_vals and len(x_vals) > 0:
                        idx = np.argsort(x_vals)
                        spectra.append(SpectrumObject(
                            id=f"PAHdb_{uid}",
                            source="PAHdb",
                            x=np.array(x_vals)[idx],
                            y=np.array(y_vals)[idx],
                            metadata={
                                "formula": formula,
                                "weight": weight,
                                "uid": uid
                            }
                        ))
                        pbar.update(1)
                    
                    # Clear element from memory
                    elem.clear()
                    
                    if limit and len(spectra) >= limit:
                        break
            pbar.close()
            print(f"Successfully parsed {len(spectra)} PAHdb spectra", flush=True)
        except Exception as e:
            print(f"Error parsing PAHdb XML {filepath}: {e}")
            import traceback
            traceback.print_exc()
        return spectra

    def parse_pds_tab(self, tab_path: str, xml_path: Optional[str] = None) -> SpectrumObject:
        """Parses PDS .tab file using its .xml label if available."""
        metadata = {}
        expected_records = None
        if xml_path and os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                metadata['title'] = root.findtext('.//{http://pds.nasa.gov/pds4/pds/v1}title')
                metadata['specimen_name'] = root.findtext('.//{http://pds.nasa.gov/pds4/speclib/v1}specimen_name')
                metadata['unit'] = root.findtext('.//{http://pds.nasa.gov/pds4/speclib/v1}spectral_range_unit_name')
                records_text = root.findtext('.//{http://pds.nasa.gov/pds4/pds/v1}records')
                if records_text:
                    expected_records = int(records_text)
            except:
                pass

        try:
            with open(tab_path, 'r') as f:
                lines = f.readlines()
                if not lines: return None
                
                first_line = lines[0].strip()
                start_idx = 0
                if first_line.isdigit():
                    expected_records = int(first_line)
                    start_idx = 1
                
                # Filter only lines that look like data (start with numbers)
                data_lines = []
                for i in range(start_idx, len(lines)):
                    line = lines[i].strip()
                    if not line: continue
                    # Simple check if line starts with a number
                    if any(c.isdigit() for c in line.split()[0]):
                        data_lines.append(line)
                    
                    if expected_records and len(data_lines) >= expected_records:
                        break
                
                # Process data_lines
                data = [list(map(float, line.split()[:2])) for line in data_lines if len(line.split()) >= 2]
                if not data: return None
                
                df = pd.DataFrame(data, columns=['x', 'y'])
            
            x = df['x'].values
            y = df['y'].values
            
            unit = metadata.get('unit', '').lower()
            if 'nm' in unit or (x.max() > 1000):
                # Convert nm to cm-1 (wavenumbers) - avoid divide by zero
                x = 10000000.0 / (x + 1e-9)
            elif 'micrometer' in unit or 'um' in unit or (x.max() < 50 and x.max() > 0.1):
                # Convert um to cm-1 - avoid divide by zero
                x = 10000.0 / (x + 1e-9)
            
            idx = np.argsort(x)
            
            return SpectrumObject(
                id=os.path.basename(tab_path),
                source="PDS",
                x=x[idx],
                y=y[idx],
                metadata=metadata
            )
        except Exception as e:
            print(f"Error parsing PDS TAB {tab_path}: {e}")
            return None

    def scan_directory(self, root_dir: str, pahdb_limit: Optional[int] = 100) -> List[SpectrumObject]:
        """Recursively scans directory and parses recognized spectral files."""
        all_spectra = []
        print(f"Scanning directory: {root_dir}", flush=True)
        
        # Phase 1: Collect files
        walk_list = list(os.walk(root_dir))
        pbar = tqdm(desc="Scanning files", unit="file")
        
        for root, dirs, files in walk_list:
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if file == "pahdb-complete-theoretical-v4.00.xml":
                    print(f"\nFound PAHdb database. Parsing...", flush=True)
                    all_spectra.extend(self.parse_pahdb_xml(filepath, limit=pahdb_limit))
                elif ext == ".tab":
                    xml_path = filepath.replace(".tab", ".xml")
                    spectra = self.parse_pds_tab(filepath, xml_path if os.path.exists(xml_path) else None)
                    if spectra:
                        all_spectra.append(spectra)
                pbar.update(1)
        pbar.close()
        print(f"Scanning complete. Total spectra collected: {len(all_spectra)}", flush=True)
        return all_spectra

# Alias for backwards compatibility
Spectrum = SpectrumObject

if __name__ == "__main__":
    parser = UniversalParser()
