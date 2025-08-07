#!/usr/bin/env python3
"""
Parameter Mapper: Bridge between Pedalboard and Differentiable Audio Tools

This module handles parameter mapping and compatibility between:
1. Real pedalboard parameters (non-differentiable)
2. Differentiable audio proxy parameters (for training)

Key Features:
- Parameter name mapping and normalization
- Range conversion between different parameter systems
- Compatibility checking and filtering
- Fallback handling for unsupported parameters
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

class ParameterMapper:
    """
    Maps and converts parameters between pedalboard and differentiable proxies
    """
    
    def __init__(self):
        # Define parameter mappings between systems
        self.parameter_mappings = {
            # Equalizer mappings
            'equalizer': {
                'pedalboard_to_diff': {
                    'Frequency': 'center_freq',  # Hz -> Hz
                    'frequency': 'center_freq',  # Hz -> Hz (alternative case)  
                    'Gain': 'gain_db',          # dB -> dB  
                    'Q': 'q',                   # Q factor -> Q factor
                    'Filter-type': 'filter_type' # Now supported!
                },
                'diff_to_pedalboard': {
                    'center_freq': 'Frequency',
                    'gain_db': 'Gain',
                    'q': 'Q',
                    'filter_type': 'Filter-type'
                },
                'supported_in_diff': ['center_freq', 'gain_db', 'q', 'filter_type'],
                'supported_in_pedalboard': ['Frequency', 'frequency', 'Gain', 'Q', 'Filter-type'],
                # Value mappings for filter types
                'value_mappings': {
                    'Filter-type': {
                        # Pedalboard -> Differentiable
                        'bell': 'bell',
                        'peak': 'peak', 
                        'highpass': 'highpass',
                        'lowpass': 'lowpass',
                        'high-shelf': 'highshelf',
                        'low-shelf': 'lowshelf',
                        'notch': 'notch',
                        'bandpass': 'bell',  # Fallback to bell
                    },
                    'filter_type': {
                        # Differentiable -> Pedalboard  
                        'bell': 'bell',
                        'peak': 'peak',
                        'highpass': 'highpass', 
                        'lowpass': 'lowpass',
                        'highshelf': 'high-shelf',
                        'lowshelf': 'low-shelf',
                        'notch': 'notch'
                    }
                }
            },
            
            # Reverb mappings
            'reverb': {
                'pedalboard_to_diff': {
                    'Room Size': 'room_size',    # 0-10 -> 0-1 (needs scaling)
                    'Pre Delay': 'pre_delay',    # seconds -> seconds
                    'Diffusion': 'diffusion',    # 0-1 -> 0-1
                    'Damping': 'damping',        # 0-1 -> 0-1
                    'Wet Gain': 'wet_gain'       # 0-1 -> 0-1
                },
                'diff_to_pedalboard': {
                    'room_size': 'Room Size',
                    'pre_delay': 'Pre Delay',
                    'diffusion': 'Diffusion',
                    'damping': 'Damping',
                    'wet_gain': 'Wet Gain'
                },
                'supported_in_diff': ['room_size', 'pre_delay', 'diffusion', 'damping', 'wet_gain'],
                'supported_in_pedalboard': ['Room Size', 'Pre Delay', 'Diffusion', 'Damping', 'Wet Gain'],
                # Range conversions
                'range_conversions': {
                    'Room Size': {'from': (0, 10), 'to': (0, 1)},  # pedalboard 0-10 -> diff 0-1
                    'room_size': {'from': (0, 1), 'to': (0, 10)}   # diff 0-1 -> pedalboard 0-10
                }
            },
            
            # Distortion mappings
            'distortion': {
                'pedalboard_to_diff': {
                    'Gain': 'gain',      # dB -> linear gain (needs conversion)
                    'Color': 'color'     # 0-1 -> 0-1
                },
                'diff_to_pedalboard': {
                    'gain': 'Gain',
                    'color': 'Color'
                },
                'supported_in_diff': ['gain', 'color'],
                'supported_in_pedalboard': ['Gain', 'Color'],
                'range_conversions': {
                    'Gain': {'from': (0, 30), 'to': (1, 10)},     # dB -> linear
                    'gain': {'from': (1, 10), 'to': (0, 30)}      # linear -> dB
                }
            },
            
            # Pitch mappings
            'pitch': {
                'pedalboard_to_diff': {
                    'Scale': 'pitch_shift'  # semitones -> semitones
                },
                'diff_to_pedalboard': {
                    'pitch_shift': 'Scale'
                },
                'supported_in_diff': ['pitch_shift'],
                'supported_in_pedalboard': ['Scale']
            }
        }
    
    def map_pedalboard_to_diff(self, pedalboard_params: Dict) -> Dict:
        """
        Convert pedalboard parameters to differentiable proxy parameters
        
        Args:
            pedalboard_params: Pedalboard format parameters
            
        Returns:
            diff_params: Differentiable proxy format parameters
        """
        diff_params = {}
        
        for effect_name, effect_params in pedalboard_params.items():
            effect_key = effect_name.lower()
            
            if effect_key not in self.parameter_mappings:
                warnings.warn(f"Effect '{effect_name}' not supported in parameter mapping")
                continue
                
            mapping = self.parameter_mappings[effect_key]
            diff_effect_params = {}
            
            if effect_name == 'Equalizer':
                # Handle EQ bands specially
                for band_id, band_params in effect_params.items():
                    diff_band_params = self._map_single_effect_params(
                        band_params, mapping, 'pedalboard_to_diff'
                    )
                    if diff_band_params:
                        diff_effect_params[band_id] = diff_band_params
            else:
                # Handle other effects
                diff_effect_params = self._map_single_effect_params(
                    effect_params, mapping, 'pedalboard_to_diff'
                )
            
            if diff_effect_params:
                diff_params[effect_key] = diff_effect_params
        
        return diff_params
    
    def map_diff_to_pedalboard(self, diff_params: Dict) -> Dict:
        """
        Convert differentiable proxy parameters to pedalboard parameters
        
        Args:
            diff_params: Differentiable proxy format parameters
            
        Returns:
            pedalboard_params: Pedalboard format parameters
        """
        pedalboard_params = {}
        
        for effect_name, effect_params in diff_params.items():
            if effect_name not in self.parameter_mappings:
                continue
                
            mapping = self.parameter_mappings[effect_name]
            
            if effect_name == 'equalizer':
                # Handle EQ bands specially
                pedalboard_eq_params = {}
                for band_id, band_params in effect_params.items():
                    pedalboard_band_params = self._map_single_effect_params(
                        band_params, mapping, 'diff_to_pedalboard'
                    )
                    if pedalboard_band_params:
                        # Add default filter type if not present
                        if 'Filter-type' not in pedalboard_band_params:
                            pedalboard_band_params['Filter-type'] = 'bell'
                        pedalboard_eq_params[band_id] = pedalboard_band_params
                
                if pedalboard_eq_params:
                    pedalboard_params['Equalizer'] = pedalboard_eq_params
            else:
                # Handle other effects
                effect_title = effect_name.title()
                pedalboard_effect_params = self._map_single_effect_params(
                    effect_params, mapping, 'diff_to_pedalboard'
                )
                if pedalboard_effect_params:
                    pedalboard_params[effect_title] = pedalboard_effect_params
        
        return pedalboard_params
    
    def _map_single_effect_params(self, params: Dict, mapping: Dict, direction: str) -> Dict:
        """Map parameters for a single effect"""
        mapped_params = {}
        param_mapping = mapping[direction]
        range_conversions = mapping.get('range_conversions', {})
        value_mappings = mapping.get('value_mappings', {})
        
        for old_key, value in params.items():
            new_key = param_mapping.get(old_key)
            
            if new_key is None:
                # Parameter not supported in target system - collect for batch warning
                if not hasattr(self, '_unsupported_params_warned'):
                    self._unsupported_params_warned = set()
                
                # Only warn once per parameter type
                if old_key not in self._unsupported_params_warned:
                    warnings.warn(f"Parameter '{old_key}' not supported in target system")
                    self._unsupported_params_warned.add(old_key)
                continue
            
            # Handle value mappings (e.g., filter types)
            if old_key in value_mappings:
                converted_value = self._convert_parameter_value_mapping(
                    old_key, value, value_mappings[old_key]
                )
            else:
                # Convert numeric value if needed
                converted_value = self._convert_parameter_value(
                    old_key, value, range_conversions
                )
            
            mapped_params[new_key] = converted_value
        
        return mapped_params
    
    def _convert_parameter_value_mapping(self, param_name: str, value: Any, value_map: Dict) -> Any:
        """Convert parameter value using value mapping (for non-numeric parameters)"""
        if isinstance(value, str):
            # String value mapping
            mapped_value = value_map.get(value.lower(), value)  # Case-insensitive lookup
            return mapped_value
        else:
            # For non-string values, return as-is
            return value
    
    def _convert_parameter_value(self, param_name: str, value: Any, 
                                range_conversions: Dict) -> Any:
        """Convert parameter value based on range conversion rules"""
        if param_name not in range_conversions:
            return value
        
        conversion = range_conversions[param_name]
        from_range = conversion['from']
        to_range = conversion['to']
        
        # Convert tensor to scalar if needed
        if isinstance(value, torch.Tensor):
            scalar_value = value.detach().cpu().item()
        else:
            scalar_value = float(value)
        
        # Linear scaling
        from_min, from_max = from_range
        to_min, to_max = to_range
        
        # Normalize to 0-1
        normalized = (scalar_value - from_min) / (from_max - from_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Scale to target range
        converted = to_min + normalized * (to_max - to_min)
        
        # Return same type as input
        if isinstance(value, torch.Tensor):
            return torch.tensor(converted).to(value.device)
        else:
            return converted
    
    def get_supported_parameters(self, effect_name: str, system: str = 'both') -> Dict:
        """
        Get supported parameters for an effect in different systems
        
        Args:
            effect_name: Name of the effect
            system: 'pedalboard', 'diff', or 'both'
            
        Returns:
            supported_params: Dictionary of supported parameters
        """
        effect_key = effect_name.lower()
        
        if effect_key not in self.parameter_mappings:
            return {}
        
        mapping = self.parameter_mappings[effect_key]
        
        if system == 'pedalboard':
            return {'supported': mapping['supported_in_pedalboard']}
        elif system == 'diff':
            return {'supported': mapping['supported_in_diff']}
        else:  # both
            return {
                'pedalboard': mapping['supported_in_pedalboard'],
                'differentiable': mapping['supported_in_diff'],
                'common': list(set(mapping['pedalboard_to_diff'].keys()) & 
                              set(mapping['diff_to_pedalboard'].keys()))
            }
    
    def check_compatibility(self, pedalboard_params: Dict) -> Dict:
        """
        Check compatibility between pedalboard and differentiable parameters
        
        Args:
            pedalboard_params: Pedalboard parameters to check
            
        Returns:
            compatibility_report: Report of compatibility status
        """
        report = {
            'supported_effects': {},
            'unsupported_effects': {},
            'missing_in_diff': {},
            'warnings': [],
            'overall_compatible': True
        }
        
        for effect_name, effect_params in pedalboard_params.items():
            effect_key = effect_name.lower()
            
            if effect_key not in self.parameter_mappings:
                report['unsupported_effects'][effect_name] = f"Effect not supported"
                report['overall_compatible'] = False
                continue
            
            mapping = self.parameter_mappings[effect_key]
            supported_in_diff = set(mapping['supported_in_diff'])
            
            if effect_name == 'Equalizer':
                # Check EQ bands
                for band_id, band_params in effect_params.items():
                    compatible_params = []
                    incompatible_params = []
                    
                    for param_name in band_params.keys():
                        diff_param = mapping['pedalboard_to_diff'].get(param_name)
                        if diff_param and diff_param in supported_in_diff:
                            compatible_params.append(param_name)
                        else:
                            incompatible_params.append(param_name)
                    
                    if compatible_params:
                        band_key = f"{effect_name}_band_{band_id}"
                        report['supported_effects'][band_key] = compatible_params
                    if incompatible_params:
                        band_key = f"{effect_name}_band_{band_id}"
                        report['missing_in_diff'][band_key] = incompatible_params
                        report['overall_compatible'] = False
            else:
                # Check other effects
                compatible_params = []
                incompatible_params = []
                
                for param_name in effect_params.keys():
                    diff_param = mapping['pedalboard_to_diff'].get(param_name)
                    if diff_param and diff_param in supported_in_diff:
                        compatible_params.append(param_name)
                    else:
                        incompatible_params.append(param_name)
                
                if compatible_params:
                    report['supported_effects'][effect_name] = compatible_params
                if incompatible_params:
                    report['missing_in_diff'][effect_name] = incompatible_params
                    report['overall_compatible'] = False
        
        # Add warnings for missing functionality
        if report['missing_in_diff']:
            report['warnings'].append(
                "Some parameters will be ignored during differentiable training"
            )
        
        return report


def test_parameter_mapper():
    """Test the parameter mapper functionality"""
    
    print("🔧 Testing Parameter Mapper")
    print("=" * 40)
    
    mapper = ParameterMapper()
    
    # Example pedalboard parameters
    pedalboard_params = {
        "Equalizer": {
            1: {"frequency": 1000, "Gain": 5, "Q": 1.0, "Filter-type": "bell"},
            2: {"frequency": 3500, "Gain": -3, "Q": 0.7, "Filter-type": "high-shelf"}
        },
        "Reverb": {
            "Room Size": 9,
            "Pre Delay": 0.1,
            "Diffusion": 0.7,
            "Damping": 0.4,
            "Wet Gain": 0.8
        },
        "Distortion": {
            "Gain": 15,
            "Color": 0.8
        },
        "Pitch": {
            "Scale": -7
        }
    }
    
    print("📊 Original Pedalboard Parameters:")
    for effect, params in pedalboard_params.items():
        print(f"  {effect}: {params}")
    
    # Test compatibility check
    print("\n🔍 Compatibility Check:")
    compatibility = mapper.check_compatibility(pedalboard_params)
    
    print("✅ Compatible parameters:")
    for effect, params in compatibility['compatible'].items():
        print(f"  {effect}: {params}")
    
    print("❌ Missing in differentiable version:")
    for effect, params in compatibility['missing_in_diff'].items():
        print(f"  {effect}: {params}")
    
    if compatibility['warnings']:
        print("⚠️ Warnings:")
        for warning in compatibility['warnings']:
            print(f"  {warning}")
    
    # Test parameter mapping
    print("\n🔄 Mapping to Differentiable Format:")
    diff_params = mapper.map_pedalboard_to_diff(pedalboard_params)
    
    for effect, params in diff_params.items():
        print(f"  {effect}: {params}")
    
    # Test reverse mapping
    print("\n🔄 Mapping back to Pedalboard Format:")
    reconstructed_params = mapper.map_diff_to_pedalboard(diff_params)
    
    for effect, params in reconstructed_params.items():
        print(f"  {effect}: {params}")
    
    print("\n✅ Parameter mapping test completed!")


if __name__ == "__main__":
    test_parameter_mapper()
