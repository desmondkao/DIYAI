"""
Power monitoring module for AI workloads
Provides hardware-level monitoring for CPU and GPU with estimation fallbacks
"""

import os
import time
import json
import platform
import threading
import statistics
from datetime import datetime

# Global monitoring state
_monitor = None
_is_monitoring = False
_log_file = "power_measurements.json"

# Constants for energy estimation (backup values)
# These values will be used when hardware monitoring isn't available
DEFAULT_VALUES = {
    "cpu": {
        "tdp_watts": 65,  # Average TDP for mid-range CPU
        "idle_watts": 15,  # Idle power consumption
    },
    "gpu": {
        "tdp_watts": 150,  # TDP for mid-range GPU
        "idle_watts": 30,  # Idle power consumption
    }
}

# CO2 emissions per kWh - updated for 2025
CO2_PER_KWH = {
    "global_avg": 0.442,  # kg CO2 per kWh (global average, 2025)
    "eu": 0.198,          # kg CO2 per kWh (EU average, 2025)
    "us": 0.382,          # kg CO2 per kWh (US average, 2025)
    "china": 0.529,       # kg CO2 per kWh (China average, 2025)
    "india": 0.682        # kg CO2 per kWh (India average, 2025)
}

# Cloud efficiency factors
CLOUD_EFFICIENCY_FACTOR = 0.58  # Cloud efficiency factor (42% more efficient than local)


def get_monitor():
    """Get or create the power monitor singleton"""
    global _monitor
    if _monitor is None:
        _monitor = PowerMonitor()
    return _monitor


class PowerMonitor:
    """
    Monitors power consumption of CPU and GPU during AI workloads.
    Falls back to estimation when hardware monitoring isn't available.
    """
    
    def __init__(self):
        self.hardware_monitoring = False
        self.measurements = []
        self.start_time = None
        self.end_time = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Try to import hardware monitoring libraries
        try:
            import psutil
            self.psutil = psutil
            self._has_psutil = True
        except ImportError:
            self._has_psutil = False
            print("psutil not available. CPU monitoring will use estimation.")
        
        # Try to import GPU monitoring libraries
        self._has_gpu = False
        self._gpu_lib = None
        
        # Check for NVIDIA GPU with pynvml
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._has_gpu = True
                self._gpu_lib = "pynvml"
                self.gpu_devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            pynvml.nvmlShutdown()
        except (ImportError, Exception) as e:
            pass
        
        # If no NVIDIA GPU, try AMD GPU with pyamdgpuinfo
        if not self._has_gpu:
            try:
                import pyamdgpuinfo
                self.pyamdgpuinfo = pyamdgpuinfo
                self._has_gpu = True
                self._gpu_lib = "pyamdgpuinfo"
                self.gpu_devices = list(range(pyamdgpuinfo.detect_gpus()))
            except (ImportError, Exception) as e:
                pass
        
        # As another fallback, try gpustat which is a higher-level wrapper
        if not self._has_gpu:
            try:
                import gpustat
                self.gpustat = gpustat
                stats = gpustat.GPUStatCollection.new_query()
                if len(stats.gpus) > 0:
                    self._has_gpu = True
                    self._gpu_lib = "gpustat"
                    self.gpu_devices = list(range(len(stats.gpus)))
            except (ImportError, Exception) as e:
                pass
        
        # Set hardware monitoring flag
        self.hardware_monitoring = self._has_psutil or self._has_gpu
        
        # Get system info
        self.system_info = self._get_system_info()
    
    def _get_system_info(self):
        """Get system information"""
        info = {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu": {
                "cores": os.cpu_count(),
                "model": "Unknown",
            },
            "gpu": {
                "available": self._has_gpu,
                "library": self._gpu_lib,
                "devices": []
            }
        }
        
        # Get CPU model
        if self._has_psutil:
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                info["cpu"]["model"] = cpu_info.get("brand_raw", "Unknown")
            except ImportError:
                # Try with platform module as fallback
                if platform.system() == "Linux":
                    try:
                        with open("/proc/cpuinfo", "r") as f:
                            for line in f:
                                if "model name" in line:
                                    info["cpu"]["model"] = line.split(":")[1].strip()
                                    break
                    except:
                        pass
                elif platform.system() == "Darwin":  # macOS
                    try:
                        import subprocess
                        output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                        info["cpu"]["model"] = output
                    except:
                        pass
                elif platform.system() == "Windows":
                    try:
                        import subprocess
                        output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().strip()
                        lines = output.split("\n")
                        if len(lines) > 1:
                            info["cpu"]["model"] = lines[1].strip()
                    except:
                        pass
        
        # Get GPU info
        if self._has_gpu:
            if self._gpu_lib == "pynvml":
                try:
                    self.pynvml.nvmlInit()
                    for i, handle in enumerate(self.gpu_devices):
                        device_info = {
                            "index": i,
                            "name": self.pynvml.nvmlDeviceGetName(handle).decode(),
                            "memory_total": self.pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)  # MB
                        }
                        info["gpu"]["devices"].append(device_info)
                    self.pynvml.nvmlShutdown()
                except Exception as e:
                    pass
            
            elif self._gpu_lib == "pyamdgpuinfo":
                try:
                    for i in self.gpu_devices:
                        gpu = self.pyamdgpuinfo.get_gpu(i)
                        device_info = {
                            "index": i,
                            "name": gpu.name,
                            "memory_total": gpu.memory_info["vram_size"] / (1024 * 1024)  # MB
                        }
                        info["gpu"]["devices"].append(device_info)
                except Exception as e:
                    pass
            
            elif self._gpu_lib == "gpustat":
                try:
                    stats = self.gpustat.GPUStatCollection.new_query()
                    for i, gpu in enumerate(stats.gpus):
                        device_info = {
                            "index": i,
                            "name": gpu.name,
                            "memory_total": gpu.memory_total
                        }
                        info["gpu"]["devices"].append(device_info)
                except Exception as e:
                    pass
        
        return info
    
    def _measure_power(self):
        """
        Measure power consumption of CPU and GPU
        Returns a dictionary with measurement results
        """
        measurement = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "system": self.system_info
        }
        
        # CPU measurements
        if self._has_psutil:
            try:
                # Get CPU utilization
                cpu_percent = self.psutil.cpu_percent(interval=0.1)
                
                # Get CPU frequency
                cpu_freq = self.psutil.cpu_freq()
                
                # Get CPU temperature if available
                cpu_temp = None
                if hasattr(self.psutil, "sensors_temperatures"):
                    temps = self.psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if name.lower() in ['coretemp', 'k10temp', 'cpu_thermal']:
                                if entries:
                                    cpu_temp = statistics.mean([entry.current for entry in entries])
                                    break
                
                # Estimate CPU power based on TDP and utilization
                # This is a simple model: power = idle_power + (tdp - idle_power) * (utilization/100)
                max_power = DEFAULT_VALUES["cpu"]["tdp_watts"]
                idle_power = DEFAULT_VALUES["cpu"]["idle_watts"]
                estimated_power = idle_power + (max_power - idle_power) * (cpu_percent / 100)
                
                # Package CPU data
                measurement["cpu"] = {
                    "percent": cpu_percent,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "temperature_c": cpu_temp,
                    "estimated_power_watts": estimated_power
                }
            except Exception as e:
                # Fallback to estimation
                measurement["cpu"] = {
                    "percent": 0,
                    "estimated_power_watts": DEFAULT_VALUES["cpu"]["idle_watts"]
                }
        else:
            # No hardware monitoring - use default values
            measurement["cpu"] = {
                "percent": 0,
                "estimated_power_watts": DEFAULT_VALUES["cpu"]["idle_watts"]
            }
        
        # GPU measurements
        if self._has_gpu:
            # Initialize GPU data structure
            measurement["gpu"] = {
                "devices": []
            }
            
            if self._gpu_lib == "pynvml":
                try:
                    self.pynvml.nvmlInit()
                    for i, handle in enumerate(self.gpu_devices):
                        # Get GPU utilization
                        utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = utilization.gpu
                        
                        # Get GPU power
                        power = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # convert from mW to W
                        
                        # Get GPU memory
                        memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_used = memory.used / (1024 * 1024)  # MB
                        memory_total = memory.total / (1024 * 1024)  # MB
                        
                        # Get GPU temperature
                        temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                        
                        # Add to devices
                        measurement["gpu"]["devices"].append({
                            "index": i,
                            "utilization_gpu": gpu_util,
                            "power_watts": power,
                            "memory_used_mb": memory_used,
                            "memory_total_mb": memory_total,
                            "temperature_c": temperature
                        })
                    
                    # Calculate average values across all GPUs
                    if measurement["gpu"]["devices"]:
                        measurement["gpu"]["average_utilization"] = statistics.mean(
                            [dev["utilization_gpu"] for dev in measurement["gpu"]["devices"]]
                        )
                        measurement["gpu"]["average_power_watts"] = statistics.mean(
                            [dev["power_watts"] for dev in measurement["gpu"]["devices"]]
                        )
                        measurement["gpu"]["average_temperature"] = statistics.mean(
                            [dev["temperature_c"] for dev in measurement["gpu"]["devices"]]
                        )
                    
                    self.pynvml.nvmlShutdown()
                
                except Exception as e:
                    # Fallback to estimation
                    for i, handle in enumerate(self.gpu_devices):
                        measurement["gpu"]["devices"].append({
                            "index": i,
                            "utilization_gpu": 0,
                            "estimated_power_watts": DEFAULT_VALUES["gpu"]["idle_watts"]
                        })
            
            elif self._gpu_lib == "pyamdgpuinfo":
                try:
                    for i in self.gpu_devices:
                        gpu = self.pyamdgpuinfo.get_gpu(i)
                        
                        # Get GPU utilization
                        activity = gpu.query_activity()
                        gpu_util = activity.get("GPU Utilization (%)", 0)
                        
                        # Get GPU memory
                        memory_info = gpu.memory_info
                        memory_used = memory_info.get("vram_used", 0) / (1024 * 1024)  # MB
                        memory_total = memory_info.get("vram_size", 0) / (1024 * 1024)  # MB
                        
                        # Get GPU temperature
                        temperature = gpu.query_temperature()
                        
                        # Estimate power based on TDP and utilization
                        max_power = DEFAULT_VALUES["gpu"]["tdp_watts"]
                        idle_power = DEFAULT_VALUES["gpu"]["idle_watts"]
                        estimated_power = idle_power + (max_power - idle_power) * (gpu_util / 100)
                        
                        # Add to devices
                        measurement["gpu"]["devices"].append({
                            "index": i,
                            "utilization_gpu": gpu_util,
                            "estimated_power_watts": estimated_power,
                            "memory_used_mb": memory_used,
                            "memory_total_mb": memory_total,
                            "temperature_c": temperature
                        })
                    
                    # Calculate average values across all GPUs
                    if measurement["gpu"]["devices"]:
                        measurement["gpu"]["average_utilization"] = statistics.mean(
                            [dev["utilization_gpu"] for dev in measurement["gpu"]["devices"]]
                        )
                        measurement["gpu"]["average_estimated_power_watts"] = statistics.mean(
                            [dev["estimated_power_watts"] for dev in measurement["gpu"]["devices"]]
                        )
                        measurement["gpu"]["average_temperature"] = statistics.mean(
                            [dev["temperature_c"] for dev in measurement["gpu"]["devices"]]
                        )
                
                except Exception as e:
                    # Fallback to estimation
                    for i in self.gpu_devices:
                        measurement["gpu"]["devices"].append({
                            "index": i,
                            "utilization_gpu": 0,
                            "estimated_power_watts": DEFAULT_VALUES["gpu"]["idle_watts"]
                        })
            
            elif self._gpu_lib == "gpustat":
                try:
                    stats = self.gpustat.GPUStatCollection.new_query()
                    for i, gpu in enumerate(stats.gpus):
                        # Get GPU utilization
                        gpu_util = gpu.utilization
                        
                        # Get GPU memory
                        memory_used = gpu.memory_used
                        memory_total = gpu.memory_total
                        
                        # Get GPU temperature
                        temperature = gpu.temperature
                        
                        # Estimate power based on TDP and utilization
                        max_power = DEFAULT_VALUES["gpu"]["tdp_watts"]
                        idle_power = DEFAULT_VALUES["gpu"]["idle_watts"]
                        estimated_power = idle_power + (max_power - idle_power) * (gpu_util / 100)
                        
                        # Add to devices
                        measurement["gpu"]["devices"].append({
                            "index": i,
                            "utilization_gpu": gpu_util,
                            "estimated_power_watts": estimated_power,
                            "memory_used_mb": memory_used,
                            "memory_total_mb": memory_total,
                            "temperature_c": temperature
                        })
                    
                    # Calculate average values across all GPUs
                    if measurement["gpu"]["devices"]:
                        measurement["gpu"]["average_utilization"] = statistics.mean(
                            [dev["utilization_gpu"] for dev in measurement["gpu"]["devices"]]
                        )
                        measurement["gpu"]["average_estimated_power_watts"] = statistics.mean(
                            [dev["estimated_power_watts"] for dev in measurement["gpu"]["devices"]]
                        )
                        measurement["gpu"]["average_temperature"] = statistics.mean(
                            [dev["temperature_c"] for dev in measurement["gpu"]["devices"]]
                        )
                
                except Exception as e:
                    # Fallback to estimation
                    for i in self.gpu_devices:
                        measurement["gpu"]["devices"].append({
                            "index": i,
                            "utilization_gpu": 0,
                            "estimated_power_watts": DEFAULT_VALUES["gpu"]["idle_watts"]
                        })
        
        else:
            # No GPU monitoring - check if we should use default GPU values
            # Only add GPU estimation if the system info says a GPU is likely present
            if any(gpu_hint in str(self.system_info).lower() for gpu_hint in ["nvidia", "amd", "geforce", "radeon"]):
                measurement["gpu"] = {
                    "devices": [{
                        "index": 0,
                        "utilization_gpu": 0,
                        "estimated_power_watts": DEFAULT_VALUES["gpu"]["idle_watts"]
                    }]
                }
        
        # Calculate total estimated power
        total_power = 0
        
        # Add CPU power
        if "cpu" in measurement:
            cpu_power = measurement["cpu"].get("power_watts", 
                         measurement["cpu"].get("estimated_power_watts", 
                         DEFAULT_VALUES["cpu"]["idle_watts"]))
            total_power += cpu_power
        
        # Add GPU power
        if "gpu" in measurement:
            if "average_power_watts" in measurement["gpu"]:
                gpu_power = measurement["gpu"]["average_power_watts"]
            elif "average_estimated_power_watts" in measurement["gpu"]:
                gpu_power = measurement["gpu"]["average_estimated_power_watts"]
            elif "devices" in measurement["gpu"] and measurement["gpu"]["devices"]:
                # Take the first device as an approximation
                dev = measurement["gpu"]["devices"][0]
                gpu_power = dev.get("power_watts", dev.get("estimated_power_watts", DEFAULT_VALUES["gpu"]["idle_watts"]))
            else:
                gpu_power = DEFAULT_VALUES["gpu"]["idle_watts"]
            
            total_power += gpu_power
        
        # Add total power to measurement
        measurement["total_power_watts"] = total_power
        
        return measurement
    
    def _monitoring_loop(self):
        """Background thread for continuous monitoring"""
        interval = 0.1  # Sample every 100ms
        while not self.stop_event.is_set():
            try:
                measurement = self._measure_power()
                self.measurements.append(measurement)
            except Exception as e:
                # Log the error but continue monitoring
                print(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def start_monitoring(self):
        """Start power monitoring"""
        global _is_monitoring
        
        if _is_monitoring:
            return False
        
        self.start_time = time.time()
        self.measurements = []
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        _is_monitoring = True
        return True
    
    def stop_monitoring(self):
        """Stop power monitoring and return results"""
        global _is_monitoring
        
        if not _is_monitoring:
            return None
        
        self.end_time = time.time()
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        # Calculate duration
        duration = self.end_time - self.start_time
        
        # Calculate energy consumption
        total_energy_kwh = self._calculate_energy()
        
        # Calculate CO2 emissions
        co2_emissions = {region: total_energy_kwh * factor 
                         for region, factor in CO2_PER_KWH.items()}
        
        # Calculate cloud comparison
        cloud_energy_kwh = total_energy_kwh * CLOUD_EFFICIENCY_FACTOR
        cloud_co2 = {region: cloud_energy_kwh * factor 
                     for region, factor in CO2_PER_KWH.items()}
        
        # Create results
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "total_energy_kwh": total_energy_kwh,
            "co2_emissions_kg": co2_emissions,
            "hardware_monitoring_available": self.hardware_monitoring,
            "sample_count": len(self.measurements),
            "comparison": {
                "cloud_energy_kwh": cloud_energy_kwh,
                "cloud_co2_emissions_kg": cloud_co2,
                "savings_percentage": (1 - CLOUD_EFFICIENCY_FACTOR) * 100
            }
        }
        
        # Add hardware metrics if available
        if self.measurements:
            # Add CPU metrics
            if all("cpu" in m for m in self.measurements):
                cpu_utilization = [m["cpu"].get("percent", 0) for m in self.measurements if "percent" in m["cpu"]]
                if cpu_utilization:
                    results["cpu"] = {
                        "average_utilization": statistics.mean(cpu_utilization)
                    }
                    
                    # Add power metrics
                    if any("power_watts" in m["cpu"] for m in self.measurements):
                        cpu_power = [m["cpu"]["power_watts"] for m in self.measurements if "power_watts" in m["cpu"]]
                        results["cpu"]["average_power_watts"] = statistics.mean(cpu_power)
                    elif any("estimated_power_watts" in m["cpu"] for m in self.measurements):
                        cpu_power = [m["cpu"]["estimated_power_watts"] for m in self.measurements 
                                     if "estimated_power_watts" in m["cpu"]]
                        results["cpu"]["estimated_average_power_watts"] = statistics.mean(cpu_power)
                    
                    # Add temperature if available
                    cpu_temp = [m["cpu"].get("temperature_c", None) for m in self.measurements 
                               if "temperature_c" in m["cpu"]]
                    cpu_temp = [t for t in cpu_temp if t is not None]
                    if cpu_temp:
                        results["cpu"]["average_temperature"] = statistics.mean(cpu_temp)
            
            # Add GPU metrics
            if all("gpu" in m for m in self.measurements):
                # Check if we have direct device data
                if all("devices" in m["gpu"] and m["gpu"]["devices"] for m in self.measurements):
                    # Calculate average utilization
                    gpu_utils = []
                    for m in self.measurements:
                        for dev in m["gpu"]["devices"]:
                            if "utilization_gpu" in dev:
                                gpu_utils.append(dev["utilization_gpu"])
                    
                    if gpu_utils:
                        results["gpu"] = {"average_utilization": statistics.mean(gpu_utils)}
                        
                        # Calculate average power
                        if any(any("power_watts" in dev for dev in m["gpu"]["devices"]) 
                              for m in self.measurements if "devices" in m["gpu"]):
                            gpu_power = []
                            for m in self.measurements:
                                if "devices" in m["gpu"]:
                                    for dev in m["gpu"]["devices"]:
                                        if "power_watts" in dev:
                                            gpu_power.append(dev["power_watts"])
                            
                            if gpu_power:
                                results["gpu"]["average_power_watts"] = statistics.mean(gpu_power)
                        
                        # If no direct power measurements, use estimated power
                        elif any(any("estimated_power_watts" in dev for dev in m["gpu"]["devices"]) 
                                for m in self.measurements if "devices" in m["gpu"]):
                            gpu_power = []
                            for m in self.measurements:
                                if "devices" in m["gpu"]:
                                    for dev in m["gpu"]["devices"]:
                                        if "estimated_power_watts" in dev:
                                            gpu_power.append(dev["estimated_power_watts"])
                            
                            if gpu_power:
                                results["gpu"]["estimated_average_power_watts"] = statistics.mean(gpu_power)
                        
                        # Add temperature if available
                        gpu_temp = []
                        for m in self.measurements:
                            if "devices" in m["gpu"]:
                                for dev in m["gpu"]["devices"]:
                                    if "temperature_c" in dev:
                                        gpu_temp.append(dev["temperature_c"])
                        
                        if gpu_temp:
                            results["gpu"]["average_temperature"] = statistics.mean(gpu_temp)
                
                # If we have average values directly
                elif any("average_utilization" in m["gpu"] for m in self.measurements):
                    gpu_utils = [m["gpu"]["average_utilization"] for m in self.measurements 
                                if "average_utilization" in m["gpu"]]
                    
                    if gpu_utils:
                        results["gpu"] = {"average_utilization": statistics.mean(gpu_utils)}
                        
                        # Get power
                        if any("average_power_watts" in m["gpu"] for m in self.measurements):
                            gpu_power = [m["gpu"]["average_power_watts"] for m in self.measurements 
                                        if "average_power_watts" in m["gpu"]]
                            results["gpu"]["average_power_watts"] = statistics.mean(gpu_power)
                        elif any("average_estimated_power_watts" in m["gpu"] for m in self.measurements):
                            gpu_power = [m["gpu"]["average_estimated_power_watts"] for m in self.measurements 
                                        if "average_estimated_power_watts" in m["gpu"]]
                            results["gpu"]["estimated_average_power_watts"] = statistics.mean(gpu_power)
                        
                        # Get temperature
                        if any("average_temperature" in m["gpu"] for m in self.measurements):
                            gpu_temp = [m["gpu"]["average_temperature"] for m in self.measurements 
                                       if "average_temperature" in m["gpu"]]
                            results["gpu"]["average_temperature"] = statistics.mean(gpu_temp)
        
        # Save to log file
        try:
            if os.path.exists(_log_file):
                with open(_log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {"measurements": []}
            
            logs["measurements"].append(results)
            
            with open(_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error saving power logs: {e}")
        
        _is_monitoring = False
        return results
    
    def _calculate_energy(self):
        """Calculate energy consumption from power measurements"""
        if not self.measurements:
            return 0
        
        # Get duration in hours
        duration_hrs = (self.end_time - self.start_time) / 3600
        
        # Calculate average power
        power_readings = [m.get("total_power_watts", 0) for m in self.measurements]
        avg_power_watts = statistics.mean(power_readings) if power_readings else 0
        
        # Calculate energy in kWh
        energy_kwh = (avg_power_watts * duration_hrs) / 1000
        
        return energy_kwh


# Public functions for easy use

def start_monitoring():
    """Start power monitoring"""
    monitor = get_monitor()
    return monitor.start_monitoring()


def stop_monitoring():
    """Stop power monitoring and return results"""
    monitor = get_monitor()
    return monitor.stop_monitoring()


# Initialize on module load
if __name__ == "__main__":
    # Run a simple test if executed directly
    print("Starting power monitoring test...")
    start_monitoring()
    
    # Generate some load
    print("Generating load...")
    start_time = time.time()
    while time.time() - start_time < 5:
        # CPU load
        x = 0
        for i in range(10000000):
            x += i
    
    # Stop monitoring and print results
    results = stop_monitoring()
    print("\nMonitoring Results:")
    print(json.dumps(results, indent=2))