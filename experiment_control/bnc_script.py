"""
Example script: configure both channels of the BNC 855B-12 signal generator.

Adjust BNC_RESOURCE to match the USB address shown by:
    import pyvisa
    print(pyvisa.ResourceManager().list_resources())
"""

from bnc_control import BNC855B
import pyvisa

BNC_RESOURCE = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'

# Channel 1: 1 GHz, 0 dBm, 0° phase
CH1_FREQ_HZ   = 1e9
CH1_POWER_DBM = -10
CH1_PHASE_DEG = 0.0

# Channel 2: 1 GHz, 0 dBm, 90° phase (quadrature)
CH2_FREQ_HZ   = 1e9
CH2_POWER_DBM = -10
CH2_PHASE_DEG = 90.0

print(pyvisa.ResourceManager().list_resources())

with BNC855B(BNC_RESOURCE) as sig:
    sig.set_reference(source='INT')

    sig.configure_channel(1, CH1_FREQ_HZ, CH1_POWER_DBM, CH1_PHASE_DEG)
    sig.configure_channel(2, CH2_FREQ_HZ, CH2_POWER_DBM, CH2_PHASE_DEG)

    sig.enable_all_outputs()

    print(f"Ch1: {sig.get_frequency(1)/1e9:.6f} GHz, "
          f"{sig.get_power(1):.1f} dBm, "
          f"{sig.get_phase(1):.1f} deg, "
          f"output={'ON' if sig.get_output_state(1) else 'OFF'}")
    print(f"Ch2: {sig.get_frequency(2)/1e9:.6f} GHz, "
          f"{sig.get_power(2):.1f} dBm, "
          f"{sig.get_phase(2):.1f} deg, "
          f"output={'ON' if sig.get_output_state(2) else 'OFF'}")

    input("Press Enter to disable outputs and exit...")
