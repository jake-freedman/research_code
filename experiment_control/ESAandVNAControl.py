
import pyvisa
from RsInstrument.RsInstrument import RsInstrument
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


class ESA:
    
    def __init__(self, resource_str):
        
        self.instrument = RsInstrument(resource_str, True, True, "SelectVisa='rs'")
        
    def close(self):
        
        self.instrument.close()
        
    def set_center_freq(self, freq):
        
        self.instrument.write_str_with_opc('FREQuency:CENTer {}'.format(freq))
        
    def set_freq_span(self, span):
        
        self.instrument.write_str_with_opc("FREQuency:SPAN {}".format(span))
        
    def set_res_bw(self, bw):
        
        self.instrument.write_str_with_opc("BAND:RES {}".format(bw))
        
    def set_sweep_points(self, num_points):
        
        self.instrument.write_str_with_opc("SWeep:POINTS {}".format(num_points))
        
    def sweep_and_record(self, duration):
        
        self.instrument.write_str_with_opc("DISPlay:TRACe1:MODE MAXHold") # Not sure what this does
        self.instrument.write_str_with_opc("INITiate:CONTinuous ON") # begin continuous sweep
        sleep(duration) # wait for sweep to complete, in future this could be cleaner with single sweep
        self.instrument.write_str_with_opc("INITiate:CONTinuous OFF") # end sweep
        
        trace_power = self.instrument.query_str("Trace:DATA? TRACe1")
        trace_freq = self.instrument.query_str("TRACe:DATA:X? TRACe1")
        
        trace_power_arr = np.array([float(power) for power in trace_power.split(",")])
        trace_freq_arr = np.array([float(freq) for freq in trace_freq.split(",")])
        
        return (trace_freq_arr, trace_power_arr)
    

class VNA:
    
    def __init__(self, address):
        
        self.rm = pyvisa.ResourceManager()
        self.instrument = self.rm.open_resource(address)
        self.instrument.timeout = 20000 # max timeout time of 20 seconds
        print(f"Connected to {self.instrument.query('*IDN?')}")
        
    def set_CW_mode(self, freq, power):
        
        self.instrument.write(":SENSe1:SWEep:TYPE CW")
        self.instrument.write(":SENSe1:FREQuency {}".format(freq))
        self.instrument.write(":SOURCE1:POWer {}".format(power))
        self.instrument.write(":SENSe1:SWEep:MODE Continuous")
        
    
    def run_freq_sweep(self, freq_range, ifbw, power, num_points, data_dir, **kwargs):
        
        
        
        # self.instrument.write('CALC1:FORM MLOG') #change the format MLINear, MLOG, PHASe, UPH, IMAG, REAL,...
        self.instrument.write('SENS1:Sweep:TYPE lin')
        self.instrument.write('SENS1:FREQ:STAR {}'.format(freq_range[0]))
        self.instrument.write('SENS1:FREQ:STOP {}'.format(freq_range[1]))
        self.instrument.write('SENS1:SWE:POIN {}'.format(num_points))
        self.instrument.write('SOURCE1:POW {}'.format(power)) # this works for now but IDK the difference between SOURCE and SENS
        self.instrument.write('SENS1:BAND {}'.format(ifbw))
        
        full_ext = data_dir + '_' + self._get_dt_str() + '.s2p'
        print(full_ext)
        opc = self.instrument.query('SENS:SWE:MODE SING;*OPC?') # set trigger to single and wait till measurement is finsihed
        self.instrument.write('CALC1:MEAS1:DATA:SNP:PORTs:Save "1,2", "{}"'.format(full_ext))
        # self.instrument.write('SENS1:SWE:MODE HOLD')
        
    
    def run_s11_sweep(self, freq_range, ifbw, power, num_points, data_dir, **kwargs):
        
        # self.instrument.write('CALC1:FORM MLOG') #change the format MLINear, MLOG, PHASe, UPH, IMAG, REAL,...
        self.instrument.write('SENS1:Sweep:TYPE lin')
        self.instrument.write('SENS1:FREQ:STAR {}'.format(freq_range[0]))
        self.instrument.write('SENS1:FREQ:STOP {}'.format(freq_range[1]))
        self.instrument.write('SENS1:SWE:POIN {}'.format(num_points))
        self.instrument.write('SOURCE1:POW {}'.format(power)) # this works for now but IDK the difference between SOURCE and SENS
        self.instrument.write('SENS1:BAND {}'.format(ifbw))
        
        full_ext = os.path.join(data_dir, 's_params' + self._get_dt_str() + '.s2p')
        # print(full_ext)
        opc = self.instrument.query('SENS:SWE:MODE SING;*OPC?') # set trigger to single and wait till measurement is finsihed
        self.instrument.write('CALC1:MEAS1:DATA:SNP:PORTs:Save "1,2", "{}"'.format(full_ext))
        self.instrument.query('SENS:SWE:MODE CONT;*OPC?')
        
    def _get_dt_str(self):
        
        dt = datetime.now()
        dt_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
        
        return dt_str
    
    
    def hold(self):
        
        self.instrument.write(":SENSe1:SWEep:Mode Hold")
    
        
    def close(self):
        
        self.instrument.close()
        
        

class ESA_VNA_Joint_Measurement():
    """Class to represent a measurement involving both the ESA and VNA
    """
    
    aom_freq = 125e6
    
    def __init__(self, ESA, VNA):
        """

        Parameters
        ----------
        ESA : ESA
        VNA : VNA

        Returns
        -------
        None.

        """
        self.ESA = ESA
        self.VNA = VNA
    
    def _write_array_to_csv(self, file_name, data, metadata=None):
        """
        Parameters
        ----------
        file_name : str
            Name of file to which the method will write the data
        data : np.Array
            array containing data
        metadata : dict, optional
            dictionary containing information about the data. The default is None.

        Returns
        -------
        None.
        """
        # write data
        np.savetxt(file_name, data, delimiter=',')
        
        # write metadata
        if metadata:
            with open(file_name, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write('# Metadata:\n')
                for key, value in metadata.items():
                    f.write(f'# {key}: {value}\n')
                
                f.write(content)
    
    
                
    
    # def measure_aom_stability(self, n_iters, duration):
    #     """Basically does what vna_drive_esa_record does but without the drive
    #     and looks at the direct mixing of the aom frequency mixing with the carrier
    #     """
        
        
    #     freq_span = 50e6
    #     res_bw = 100
    #     num_points = 1001
    #     self.ESA.set_center_freq(self.aom_freq)
    #     self.ESA.set_freq_span(freq_span)
    #     self.ESA.set_res_bw(res_bw)
    #     self.ESA.set_sweep_points(num_points)
        
    #     measurement_dir =  file_name = os.path.join(self.data_dir, 'at' + self._get_dt_str())
    #     os.makedirs(measurement_dir)
        
    #     for i in range(n_iters):
            
    #         file_name = os.path.join(measurement_dir, 'at' + self._get_dt_str() + 'iter{}'.format(i))
    #         freqs, powers = self.ESA.sweep_and_record(duration)
    #         data = np.zeros((2, freqs.size))
    #         data[0,:] = freqs
    #         data[1,:] = powers
    #         metadata = {"Time": self._get_dt_str(),
    #                     "Drive Frequency": "No Drive",
    #                     "Drive Power": "No Drive",
    #                     "Resolution Bandwidth": str(res_bw),
    #                     "Description:": "characterization of the power stability of the AOM"}
            
    #         self._write_array_to_csv(file_name, data, metadata=metadata)
            


    def _get_dt_str(self):
        
        dt = datetime.now()
        dt_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
        
        return dt_str
    
    
    def vna_drive_esa_record(self, drive_freq, drive_power, center_freq, freq_span, res_bw, num_points, duration):
        
        self.VNA.set_CW_mode(drive_freq, drive_power)
        self.ESA.set_center_freq(center_freq)
        self.ESA.set_freq_span(freq_span)
        self.ESA.set_res_bw(res_bw)
        self.ESA.set_sweep_points(num_points)
        freqs, powers = self.ESA.sweep_and_record(duration)
        
        return (freqs, powers)
    
    
    def run_freq_sweep(self, drive_freqs, drive_powers, name, desc, **kwargs):
        
        default_orders = [0, 1]
        default_res_bw = 1000 # usually 1000
        default_duration = 0.15 # usually 0.15
        default_num_points = 1001
        default_data_dir = r"C:\Users\jmfreedman\OneDrive - University of Arizona\QuantumNanophoXonics\ao_chip\PhaseModulators\Data"
        default_freq_span = 20e6
        
        orders = kwargs.get('orders', default_orders)
        res_bw = kwargs.get('res_bw', default_res_bw)
        duration = kwargs.get('duration', default_duration)
        num_points = kwargs.get('num_points', default_num_points)
        data_dir = kwargs.get('data_dir', default_data_dir)
        freq_span = kwargs.get('freq_span', default_freq_span)
        
        measurement_dir = name + '_' + self._get_dt_str()
        os.makedirs(os.path.join(data_dir, measurement_dir))
        
        for order in orders:
            os.makedirs(os.path.join(data_dir, measurement_dir, 'sb_{}'.format(order)))
            
        with open(os.path.join(data_dir, measurement_dir, 'desc.txt'), 'w') as f:
            f.write(desc)
        
        for i in range(drive_freqs.size):
            
            drive_power = drive_powers[i]
            drive_freq = drive_freqs[i]
            for j in range(len(orders)):
                
                data = np.zeros(shape=(2, num_points))
                center_freq = self.aom_freq
                if orders[j] != 0:
                    center_freq = abs(orders[j])*drive_freq - np.sign(orders[j])*self.aom_freq
                
                freqs, powers = self.vna_drive_esa_record(drive_freq, 
                                                          drive_power, 
                                                          center_freq, 
                                                          freq_span, 
                                                          res_bw, 
                                                          num_points, 
                                                          duration)
                data[0, :] = freqs
                data[1, :] = powers
                
                dt_str = self._get_dt_str()
                subfolder = 'sb_{}'.format(orders[j])
                fname = 'freq{:.4f}_ind{}.csv'.format(drive_freq*1e-9, i)
                
                metadata = {'Time': dt_str,
                            'Drive Frequency': str(drive_freq),
                            'Drive Power': str(drive_power),
                            'Resolution Bandwidth': str(res_bw)}
                
                full_path = os.path.join(data_dir, measurement_dir, subfolder, fname)
                self._write_array_to_csv(full_path, data, metadata)
                
            print('Completed drive frequency {} out of {}.'.format(i+1, drive_freqs.size))
            
            
        return os.path.join(data_dir, measurement_dir)
            
        
    def close(self):
        
        self.ESA.close()
        self.VNA.close()
            
        
def write_array_to_csv_with_metadata(filename, data, metadata=None):
    # Write data to CSV file
    np.savetxt(filename, data, delimiter=',')
    
    # Write metadata as comments in the CSV file
    if metadata:
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write('# Metadata:\n')
            for key, value in metadata.items():
                f.write(f'# {key}: {value}\n')
            f.write(content)


def vna_drive_esa_record(VNA, ESA, drive_freq, drive_power, center_freq, freq_span, res_bw, num_points, duration):
    
    VNA.set_CW_mode(drive_freq, drive_power)
    ESA.set_center_freq(center_freq)
    ESA.set_freq_span(freq_span)
    ESA.set_res_bw(res_bw)
    ESA.set_sweep_points(num_points)
    freqs, powers = ESA.sweep_and_record(duration)
    
    return (freqs, powers)


def main():
    VNA_address = 'TCPIP0::HAL9000::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR'
    resource_string = "TCPIP::169.254.216.47::INSTR"
    data_dir = r"C:\Users\jmfreedman\Documents\local_data"
    # os.makedirs(data_dir)
    # fname = 's_params_' + dev_name
    vna = VNA(VNA_address)
    # esa = ESA(resource_string)
    # m = ESA_VNA_Joint_Measurement(esa, vna)
    
    
    
    # center_freq = 2.38e9
    # freq_span = 50e6
    # min_freq = center_freq - freq_span
    # max_freq = center_freq + freq_span
    # drive_freqs = np.linspace(min_freq, max_freq, 200)
    # drive_powers = 12 * np.ones(drive_freqs.size)
    # orders = [0, 1]
    
   
    # # v_min = 0.002
    # # v_max = 1.500
    # # v_arr = np.linspace(v_min, v_max, 100)
    # # drive_powers = 20 * np.log10( v_arr / np.sqrt(50e-3) ) 


    # # drive_freqs = 2.807e9 * np.ones(drive_powers.shape)
    # # orders = [0, 1, 2, 3, 4]

    
    # desc = ''
    # dev_name = 'die2-3_wg19_ultranarrow_bi_2mm'
    
    # for i in range(1):
    
        
    #     measurement_dir = m.run_freq_sweep(drive_freqs, 
    #                                         drive_powers, 
    #                                         dev_name,
    #                                         desc, 
    #                                         orders=orders, 
    #                                         data_dir=data_dir)
    
    # # vna.run_s11_sweep([min_freq, max_freq], ifbw, drive_powers[0], num_points, measurement_dir)
    # m.close()
    
    
if __name__ == "__main__":
    main()
