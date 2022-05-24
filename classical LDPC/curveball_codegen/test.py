import code_gen as cg
import numpy as np


nmult = 2
bit_deg = 3
check_deg = 4

parity_check = cg.safe_code_gen(
    bit_deg=bit_deg, check_deg=check_deg, nmult=nmult, rcmk=True)

