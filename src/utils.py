import re

def get_organic_family(formula: str) -> int:
    """
    Heuristic to categorize chemical formulas into families.
    0: Pure PAH (C, H)
    1: N-PAH (contains N)
    2: O-PAH (contains O)
    3: S-PAH (contains S)
    4: Hetero/Other
    """
    if not formula: return 4
    
    # Check for Nitrogen
    if 'N' in formula: return 1
    # Check for Oxygen
    if 'O' in formula: return 2
    # Check for Sulfur
    if 'S' in formula: return 3
    
    # Check if only C and H
    if re.fullmatch(r'C\d*H\d*', formula):
        return 0
        
    return 4

FAMILIES = {
    0: "Pure PAH",
    1: "N-PAH",
    2: "O-PAH",
    3: "S-PAH",
    4: "Complex Organic"
}
