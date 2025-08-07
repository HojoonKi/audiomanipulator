#!/usr/bin/env python3
"""
Normalized audio processing presets dataset.
Contains 472 presets in standardized format.
"""

fined_presets = [
    {
        "prompt": "Monster speaks in a large room",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 0.9, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 650, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -1, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.04,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8.0,
            "color": 0.7,
        },
        "Pitch": {
            "scale": -1.0,
        }
    },
    {
        "prompt": "Tiny fairy whispers in a glass bottle",
        "Equalizer": [
            {"frequency": 120, "gain": -6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.01,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 1.5,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 1.5,
        }
    },
    {
        "prompt": "Old robot shouts inside a metal tunnel",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.1, "filter_type": "bell"},
            {"frequency": 9000, "gain": -2, "q": 0.9, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.03,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 12.0,
            "color": 0.8,
        },
        "Pitch": {
            "scale": -2.0,
        }
    },
    {
        "prompt": "A live violin performance featuring broken effects in an anechoic chamber.",
        "Equalizer": [
            {"frequency": 300, "gain": -5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 0.8, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 2.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 60, "gain": -10, "q": 1, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 0.1,
            "pre_delay": 0.001,
            "diffusion": 0.0,
            "damping": 1.0,
            "wet_gain": 0.0,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the female sound tape-saturated in the oil refinery.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": 6, "q": 0.9, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.15,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Transform the speech of a male to sound cinematic rise in an attic.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "cinematic texture applied to saxophone in a lighthouse.",
        "Equalizer": [
            {"frequency": 200, "gain": -5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 500, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 2, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar treated with crystal clear effect as if in an attic.",
        "Equalizer": [
            {"frequency": 50, "gain": 6, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": -3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 700, "gain": 5, "q": 1, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.9, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound underwater and reverberant in the anechoic chamber.",
        "Equalizer": [
            {"frequency": 100, "gain": -10, "q": 1.0, "filter_type": "lowpass"},
            {"frequency": 300, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 700, "gain": -8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": -12, "q": 1.0, "filter_type": "bell"},
            {"frequency": 6000, "gain": -15, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.1,
            "pre_delay": 0.001,
            "diffusion": 0.0,
            "damping": 1.0,
            "wet_gain": 0.0,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A acoustic guitar layered with glassy shimmer in the stairwell.",
        "Equalizer": [
            {"frequency": 150, "gain": -4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 6000, "gain": 10, "q": 2.0, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.2, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "electro guitar played in the ice cave with a broken sound.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -6, "q": 1.5, "filter_type": "notch"},
            {"frequency": 2000, "gain": 7, "q": 2.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 2.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.0,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A saxophone layered with crystal clear in the busy intersection.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 7000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.02,
            "diffusion": 0.5,
            "damping": 0.8,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something buzzy in the dining hall.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 900, "gain": -5, "q": 2.0, "filter_type": "notch"},
            {"frequency": 3000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 7000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "resonant texture applied to flute in a hospital corridor.",
        "Equalizer": [
            {"frequency": 250, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 100, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "acoustic texture applied to saxophone in an airplane cabin.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.1, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Generate a scene with radio-like atmosphere in a radio studio.",
        "Equalizer": [
            {"frequency": 60, "gain": 7, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.2,
            "pre_delay": 0.005,
            "diffusion": 0.3,
            "damping": 0.95,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "chorused texture applied to drum set in a valley.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.15,
            "diffusion": 0.3,
            "damping": 0.4,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Transform the speech of a female to sound acoustic in a gymnasium.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.14,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin treated with metallic effect as if in a theater.",
        "Equalizer": [
            {"frequency": 150, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 6000, "gain": 10, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.3, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.0,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "glassy shimmer texture applied to keyboard in a cliff edge.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 9, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 12, "q": 2.0, "filter_type": "bell"},
            {"frequency": 15000, "gain": 8, "q": 1.2, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.08,
            "diffusion": 0.2,
            "damping": 0.6,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male is singing in a power plant, and the voice is nostalgic.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2000, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 5000, "gain": -4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "A male is singing in a planetarium, and the voice is frosty.",
        "Equalizer": [
            {"frequency": 100, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": -3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 6000, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 10, "q": 1.2, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A female is singing in a kitchen, and the voice is airy.",
        "Equalizer": [
            {"frequency": 80, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 7000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 15000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.04,
            "diffusion": 0.4,
            "damping": 0.5,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A acoustic guitar layered with reversed in the hospital corridor.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "Sound of acoustic guitar made hollow inside a library.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the organ sound smeared and reverberant in the amphitheater.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.75,
            "damping": 0.6,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of violin with cold processing in a cathedral.",
        "Equalizer": [
            {"frequency": 150, "gain": -5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 7000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male is singing in a church, and the voice is metallic.",
        "Equalizer": [
            {"frequency": 200, "gain": -6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1000, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 7000, "gain": 5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 11000, "gain": 4, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.65,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.85,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient ambient drum set heard in a large hall.",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "acoustic guitar played in the canyon with a whispering sound.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": -6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.2,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone played in the canyon with a plush sound.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute played in the opera house with an overdriven sound.",
        "Equalizer": [
            {"frequency": 150, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.1,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live flute performance featuring octave-down effects in a classroom.",
        "Equalizer": [
            {"frequency": 40, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.05,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "whispering texture applied to drum set in an auditorium.",
        "Equalizer": [
            {"frequency": 100, "gain": -8, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.75,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "alien texture applied to keyboard in an open field.",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": -4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.9, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.001,
            "diffusion": 0.1,
            "damping": 1.0,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 5,
        }
    },
    {
        "prompt": "A acoustic guitar layered with expanded in the subway platform.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A chorused performance of organ in a nightclub.",
        "Equalizer": [
            {"frequency": 100, "gain": 6, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": 5, "q": 1.1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "violin played in the mountain peak with a cold sound.",
        "Equalizer": [
            {"frequency": 150, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 900, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 7000, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 11000, "gain": 4, "q": 1, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.3,
            "pre_delay": 0.001,
            "diffusion": 0.05,
            "damping": 1.0,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of bass guitar with overdriven processing in a classroom.",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.04,
            "diffusion": 0.4,
            "damping": 0.6,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of trumpet made hollow inside a chamber.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -8, "q": 2.0, "filter_type": "notch"},
            {"frequency": 1200, "gain": -6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A conference center filled with lightly frosty sounds.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.0,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0.5,
        }
    },
    {
        "prompt": "A male whispers in the radio studio.",
        "Equalizer": [
            {"frequency": 80, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.2,
            "pre_delay": 0.005,
            "diffusion": 0.3,
            "damping": 0.95,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the saxophone sound overdriven and reverberant in the warehouse.",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "muted texture applied to trumpet in an underground tunnel.",
        "Equalizer": [
            {"frequency": 150, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": -5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A electro guitar layered with crystal clear in the cathedral.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the flute sound warm and reverberant in the swamp.",
        "Equalizer": [
            {"frequency": 120, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1800, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4500, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.8,
            "pre_delay": 0.008,
            "diffusion": 0.1,
            "damping": 0.95,
            "wet_gain": 0.03,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "cold texture applied to keyboard in a conference center.",
        "Equalizer": [
            {"frequency": 100, "gain": -7, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of piano with chorused processing in a stadium.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "Recording of bass guitar with vibrato processing in a cafeteria.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 180, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 700, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 1, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A electro guitar layered with telephone in the ice cave.",
        "Equalizer": [
            {"frequency": 100, "gain": -12, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 8, "q": 2.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 10, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 2.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": -15, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with resonant processing in a subway platform.",
        "Equalizer": [
            {"frequency": 120, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live saxophone performance featuring echoing effects in a silo.",
        "Equalizer": [
            {"frequency": 90, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.4,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "piano treated with squelchy effect as if in a hangar.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -4, "q": 1.8, "filter_type": "notch"},
            {"frequency": 1500, "gain": 6, "q": 2.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.9, "filter_type": "bell"},
            {"frequency": 10000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 14,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something cold in the bamboo grove.",
        "Equalizer": [
            {"frequency": 120, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.2,
            "pre_delay": 0.01,
            "diffusion": 0.2,
            "damping": 0.9,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female is singing in an oil refinery, and the voice is spacey.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A live piano performance featuring detuned effects in a server room.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "A female says something chorused in the library.",
        "Equalizer": [
            {"frequency": 150, "gain": 2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "Sound of piano made monster-like inside a chamber.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 800, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 7000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -7,
        }
    },
    {
        "prompt": "Pure muffled effect reminiscent of a library.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": -8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": -12, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -18, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.03,
            "diffusion": 0.3,
            "damping": 0.9,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of piano with cloudy processing in a living room.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.02,
            "diffusion": 0.4,
            "damping": 0.8,
            "wet_gain": 0.12,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A violin layered with underwater in the courtyard.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": -6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -10, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -15, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.5,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the cathedral with a detuned sound.",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 1, "q": 1.4, "filter_type": "bell"},
            {"frequency": 6000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "cold texture applied to trumpet in a train yard",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0.5,
        }
    },
    {
        "prompt": "A tuned male talks inside a city street.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "Recording of saxophone with percussive processing in an airport terminal.",
        "Equalizer": [
            {"frequency": 120, "gain": -5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 11000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "organ played in the concert hall with a hazy sound.",
        "Equalizer": [
            {"frequency": 90, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": -3, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of drum set with cloudy processing in an oil refinery.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": -4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a female with a vintage character in the attic.",
        "Equalizer": [
            {"frequency": 120, "gain": 5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live piano performance featuring pitch-shifted down effects in a hospital corridor.",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.5,
            "pre_delay": 0.11,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": -4,
        }
    },
    {
        "prompt": "A female is shouting in a laboratory with a saturated tone.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4500, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of drum set made metallic inside a theater.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 6000, "gain": 10, "q": 1.6, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "muted texture applied to bass guitar in an oil refinery.",
        "Equalizer": [
            {"frequency": 40, "gain": 7, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": -5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2500, "gain": -8, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live trumpet performance featuring fuzzed effects in a nightclub.",
        "Equalizer": [
            {"frequency": 120, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the violin sound chorused and reverberant in the courtyard.",
        "Equalizer": [
            {"frequency": 200, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Sound of violin made squelchy inside a cave.",
        "Equalizer": [
            {"frequency": 100, "gain": 5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "The aura of stuttering in the recording studio.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.01,
            "diffusion": 0.4,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a silo with alien effects.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 11000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "Sound of piano made octave-down inside a parking garage.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.11,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "A chorused performance of violin in an underwater space.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": -6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -10, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -14, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.5,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.8,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A robotic performance of acoustic guitar in a stairwell.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live flute performance featuring cold effects in an attic.",
        "Equalizer": [
            {"frequency": 80, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.3,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A female speaks in a gymnasium.",
        "Equalizer": [
            {"frequency": 80, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the male sound ring-modulated in the cathedral.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": -4, "q": 1.8, "filter_type": "notch"},
            {"frequency": 2500, "gain": 7, "q": 2.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 9, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of flute with glitchy processing in an elevator shaft.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.0,
            "pre_delay": 0.005,
            "diffusion": 0.1,
            "damping": 0.8,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "psychedelic texture applied to acoustic guitar in an ice cave.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1000, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "A breathy performance of trumpet in a church.",
        "Equalizer": [
            {"frequency": 150, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.13,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "ring-modulated texture applied to bass guitar in a dining hall.",
        "Equalizer": [
            {"frequency": 70, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": -4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.65,
            "damping": 0.55,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.85,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female is shouting in a small room with an alien tone.",
        "Equalizer": [
            {"frequency": 150, "gain": -5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 2500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 6000, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 11000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "Environmental sound with alien characteristics in an aquarium tunnel.",
        "Equalizer": [
            {"frequency": 60, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -8, "q": 2.5, "filter_type": "notch"},
            {"frequency": 1200, "gain": 12, "q": 3.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": -5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.5,
            "pre_delay": 0.13,
            "diffusion": 0.8,
            "damping": 0.7,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "Sound of acoustic guitar made formant-shifted inside a rocky shore.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1800, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.05,
            "diffusion": 0.3,
            "damping": 0.6,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A stairwell filled with lightly underwater sounds.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something noisy in the ship deck.",
        "Equalizer": [
            {"frequency": 80, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 14,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live piano performance featuring formant-shifted effects in a recording studio.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.01,
            "diffusion": 0.4,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of bass guitar with tape-saturated processing in a planetarium.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A muted performance of organ in an observatory dome.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": -6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live violin performance featuring granular effects in an airport terminal.",
        "Equalizer": [
            {"frequency": 150, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 11000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.13,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A ring-modulated soundscape in a church.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of piano made warm inside a closet.",
        "Equalizer": [
            {"frequency": 120, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.002,
            "diffusion": 0.1,
            "damping": 0.95,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "compressed texture applied to electro guitar in a train yard.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2000, "gain": -3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4500, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient noisy piano heard in a wind tunnel.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "octave-down texture applied to piano in an ice cave.",
        "Equalizer": [
            {"frequency": 40, "gain": 10, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": -4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.17,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "acoustic guitar treated with hazy effect as if in a subway tunnel.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A flute layered with acoustic in the mountain peak.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.3,
            "pre_delay": 0.001,
            "diffusion": 0.05,
            "damping": 1.0,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "compressed texture applied to electro guitar in a train yard.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2000, "gain": -3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4500, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient noisy piano heard in a wind tunnel.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "octave-down texture applied to piano in an ice cave.",
        "Equalizer": [
            {"frequency": 40, "gain": 10, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": -4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.17,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "acoustic guitar treated with hazy effect as if in a subway tunnel.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A flute layered with acoustic in the mountain peak.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.3,
            "pre_delay": 0.001,
            "diffusion": 0.05,
            "damping": 1.0,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the shopping mall with a ghostly sound.",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": -5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.14,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live flute performance featuring sidechained effects in a desert.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.2,
            "pre_delay": 0.001,
            "diffusion": 0.05,
            "damping": 1.0,
            "wet_gain": 0.01,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of flute made ghostly inside an oil refinery.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 11000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute played in the restaurant with a reversed sound.",
        "Equalizer": [
            {"frequency": 110, "gain": 4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -4,
        }
    },
    {
        "prompt": "A female whispers in the library.",
        "Equalizer": [
            {"frequency": 200, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "distorted texture applied to violin in a lighthouse.",
        "Equalizer": [
            {"frequency": 150, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.0,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live violin performance featuring pitch-shifted up effects in a restaurant.",
        "Equalizer": [
            {"frequency": 120, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "Sound of electro guitar made cinematic inside a museum.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.5,
            "pre_delay": 0.09,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of drum set with tremolo processing in a shopping mall.",
        "Equalizer": [
            {"frequency": 70, "gain": 7, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of saxophone with broken processing in a hangar.",
        "Equalizer": [
            {"frequency": 90, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": -4, "q": 1.8, "filter_type": "notch"},
            {"frequency": 3500, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound granular and reverberant in the wind tunnel.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "reverse reverb texture applied to electro guitar in an oil refinery.",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.7,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "organ treated with echoing effect as if in a busy intersection.",
        "Equalizer": [
            {"frequency": 70, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.02,
            "diffusion": 0.5,
            "damping": 0.8,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live piano performance featuring underwater effects in an oil refinery.",
        "Equalizer": [
            {"frequency": 90, "gain": 5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": -4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.13,
            "diffusion": 0.8,
            "damping": 0.8,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.5,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "bass guitar treated with reversed effect as if in a subway platform.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 7000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "A live violin performance featuring percussive effects in a theater.",
        "Equalizer": [
            {"frequency": 120, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient compressed flute heard in a stadium.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of drum set made cinematic rise inside a bus station.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.5,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A acoustic guitar layered with wobbly in the pub.",
        "Equalizer": [
            {"frequency": 120, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of flute made glassy shimmer inside an underground tunnel.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 12, "q": 2.0, "filter_type": "bell"},
            {"frequency": 15000, "gain": 10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A smoky performance of piano in a small room.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": -3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "squelchy ambience fills the stadium.",
        "Equalizer": [
            {"frequency": 100, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live saxophone performance featuring wet effects in a bamboo grove.",
        "Equalizer": [
            {"frequency": 120, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.2,
            "pre_delay": 0.01,
            "diffusion": 0.2,
            "damping": 0.9,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "trumpet played in the valley with a pitch-shifted up sound.",
        "Equalizer": [
            {"frequency": 150, "gain": 2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 11000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.15,
            "diffusion": 0.3,
            "damping": 0.4,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 5,
        }
    },
    {
        "prompt": "airport terminal ambience with plush male.",
        "Equalizer": [
            {"frequency": 100, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "muted texture applied to acoustic guitar in a closet.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": -10, "q": 1.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -15, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.002,
            "diffusion": 0.1,
            "damping": 0.95,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the trumpet sound distant and reverberant in the kitchen.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": -8, "q": 1.0, "filter_type": "bell"},
            {"frequency": 10000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.05,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male says something robotic in the factory floor.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.14,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "airy texture applied to trumpet in a train yard.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 7000, "gain": 9, "q": 1.4, "filter_type": "bell"},
            {"frequency": 15000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live bass guitar performance featuring radio-like effects in a lighthouse.",
        "Equalizer": [
            {"frequency": 80, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 10, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 8, "q": 2.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.0,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "keyboard treated with resonant effect as if in an oil refinery.",
        "Equalizer": [
            {"frequency": 100, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the closet with a tuned sound.",
        "Equalizer": [
            {"frequency": 60, "gain": 7, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 700, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.002,
            "diffusion": 0.1,
            "damping": 0.95,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "A female speaks in a warehouse.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.75,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "acoustic guitar treated with crystal clear effect as if in a factory floor.",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the drum set sound sidechained and reverberant in the recording studio.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.01,
            "diffusion": 0.4,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Background audio with muted tones set in a power plant.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -10, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone treated with monster-like effect as if in an oil refinery.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 10, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": -8, "q": 2.0, "filter_type": "notch"},
            {"frequency": 3500, "gain": 12, "q": 1.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 25,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -6,
        }
    },
    {
        "prompt": "A live bass guitar performance featuring bitcrushed effects in a recording studio.",
        "Equalizer": [
            {"frequency": 50, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.01,
            "diffusion": 0.4,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with glitchy processing in a warehouse.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": -6, "q": 2.0, "filter_type": "notch"},
            {"frequency": 4500, "gain": 9, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute treated with wobbly effect as if in a cliff edge.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.08,
            "diffusion": 0.2,
            "damping": 0.6,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient dry organ heard in a beach.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 2, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.0,
            "pre_delay": 0.02,
            "diffusion": 0.2,
            "damping": 0.8,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "keyboard played in the shopping mall with a plush sound.",
        "Equalizer": [
            {"frequency": 100, "gain": 7, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "The aura of distorted in the opera house.",
        "Equalizer": [
            {"frequency": 90, "gain": 5, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "A hollow performance of flute in an oil refinery.",
        "Equalizer": [
            {"frequency": 120, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 11000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.13,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "organ played in the cliff edge with a resonant sound.",
        "Equalizer": [
            {"frequency": 70, "gain": 7, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 800, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.08,
            "diffusion": 0.2,
            "damping": 0.6,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "electro guitar treated with saturated effect as if in a pub.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4500, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the arena with a gated sound.",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 6000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "electro guitar played in the concert hall with an untuned sound.",
        "Equalizer": [
            {"frequency": 120, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -0.3,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound vintage and reverberant in the laboratory.",
        "Equalizer": [
            {"frequency": 100, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of saxophone made muffled inside a silo.",
        "Equalizer": [
            {"frequency": 90, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": -8, "q": 1.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -15, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.8,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient resonant drum set heard in a wine cellar.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A drum set layered with bitcrushed in the underground tunnel.",
        "Equalizer": [
            {"frequency": 100, "gain": 5, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A buzzy performance of bass guitar in a lighthouse.",
        "Equalizer": [
            {"frequency": 60, "gain": 7, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 800, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of saxophone made acoustic inside a cave.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin played in the busy intersection with a radio-like sound.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.02,
            "diffusion": 0.5,
            "damping": 0.8,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of keyboard made smeared inside a chamber.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin played in the hospital corridor with a lo-fi sound.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.5,
            "pre_delay": 0.11,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "keyboard treated with formant-shifted effect as if in a silo.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1600, "gain": 9, "q": 2.0, "filter_type": "bell"},
            {"frequency": 3200, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A acoustic guitar layered with radio-like in the airplane cabin.",
        "Equalizer": [
            {"frequency": 100, "gain": -10, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "electro guitar played in the ship deck with an ambient sound.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.11,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live organ performance featuring sidechained effects in a dining hall.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "An ambient robotic organ heard in an elevator shaft.",
        "Equalizer": [
            {"frequency": 90, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 11000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.0,
            "pre_delay": 0.005,
            "diffusion": 0.1,
            "damping": 0.8,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "A female is singing in a mountain peak, and the voice is glitchy.",
        "Equalizer": [
            {"frequency": 150, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.3,
            "pre_delay": 0.001,
            "diffusion": 0.05,
            "damping": 1.0,
            "wet_gain": 0.02,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "An ambient warm saxophone heard in a pub.",
        "Equalizer": [
            {"frequency": 120, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the male sound vibrato in the chamber.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a garage with tape-saturated effects.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 1, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.07,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the female sound bitcrushed in the oil refinery.",
        "Equalizer": [
            {"frequency": 120, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of electro guitar with sparkling processing in a beach.",
        "Equalizer": [
            {"frequency": 100, "gain": -2.0, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4.0, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6.0, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 9.0, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8.0, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3.0,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0.0,
        }
    },
    {
        "prompt": "An ambient cinematic rise violin heard in a bus station.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.5,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the male sound vibrato in the chamber.",
        "Equalizer": [
            {"frequency": 120, "gain": 3, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 5,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a garage with tape-saturated effects.",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": -3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.12,
            "diffusion": 0.65,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the female sound bitcrushed in the oil refinery.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of electro guitar with sparkling processing in a beach.",
        "Equalizer": [
            {"frequency": 90, "gain": -2, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.0,
            "pre_delay": 0.02,
            "diffusion": 0.2,
            "damping": 0.8,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient cinematic rise violin heard in a bus station.",
        "Equalizer": [
            {"frequency": 120, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 2.0, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "A electro guitar layered with pulsing in the bathroom",
        "Equalizer": [
            {"frequency": 100, "gain": 6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 0.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.05,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of trumpet with resonant processing in a conference room",
        "Equalizer": [
            {"frequency": 80, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": -1, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of flute made glitchy inside a bar",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 0.8, "filter_type": "low-shelf"},
            {"frequency": 550, "gain": 3, "q": 2.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 3.0, "filter_type": "notch"},
            {"frequency": 12000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.02,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "percussive texture applied to keyboard in a valley",
        "Equalizer": [
            {"frequency": 90, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 2.0, "filter_type": "bell"},
            {"frequency": 9000, "gain": -1, "q": 0.9, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.07,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of violin with pitch-shifted down processing in a cave",
        "Equalizer": [
            {"frequency": 70, "gain": -3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 900, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.9,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": -5,
        }
    },
    {
        "prompt": "A recording of a male with a crystal clear character in the concert hall.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A distorted performance of keyboard in a bathroom.",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.2,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.3,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live piano performance featuring shimmering effects in a subway platform.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 7000, "gain": 9, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin treated with glowing effect as if in an arena.",
        "Equalizer": [
            {"frequency": 150, "gain": 1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of organ with whispering processing in a cafeteria.",
        "Equalizer": [
            {"frequency": 60, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": -4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of flute made ring-modulated inside a cave.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 2.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 10, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.17,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone treated with alien effect as if in a ship deck.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -6, "q": 2.0, "filter_type": "notch"},
            {"frequency": 1200, "gain": 12, "q": 2.5, "filter_type": "bell"},
            {"frequency": 3500, "gain": 9, "q": 1.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 14,
            "color": 0.8,
        },
        "Pitch": {
            "scale": -4,
        }
    },
    {
        "prompt": "Make the female sound percussive in the planetarium.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 9, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of flute made tape-saturated inside a mountain peak.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "The aura of lo-fi in the desert.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "drum set treated with psychedelic effect as if in a planetarium",
        "Equalizer": [
            {"frequency": 60, "gain": 5, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 0.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live acoustic guitar performance featuring distant effects in a busy intersection",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3200, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 10000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.1,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of drum set made formant-shifted inside an observatory dome",
        "Equalizer": [
            {"frequency": 70, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 4000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 14000, "gain": -3, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.85,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 7,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a theater with formant-shifted effects",
        "Equalizer": [
            {"frequency": 50, "gain": 3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": -3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": 5, "q": 2.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": -1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.9,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 5,
        }
    },
    {
        "prompt": "A male whispers in the library",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 2500, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.8,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of drum set with untuned processing in a cathedral",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "An ambient stuttering saxophone heard in a church",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.85,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the airplane cabin with a breathy sound",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": -1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3000, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 0.9, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.02,
            "diffusion": 0.4,
            "damping": 0.8,
            "wet_gain": 0.25,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live saxophone performance featuring warm effects in a restaurant",
        "Equalizer": [
            {"frequency": 80, "gain": 1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": -1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A monster-like performance of flute in a chamber",
        "Equalizer": [
            {"frequency": 50, "gain": 10, "q": 1.5, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": -4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": 8, "q": 2.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": -3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -8,
        }
    },
    {
        "prompt": "Recording of piano with pitch-shifted up processing in a train yard.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "Make the female sound overdriven in the arena.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 9, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 22,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "untuned texture applied to bass guitar in a train station.",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 1, "q": 1.4, "filter_type": "bell"},
            {"frequency": 6000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -0.5,
        }
    },
    {
        "prompt": "An ambient sparkling keyboard heard in a basement.",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 7000, "gain": 9, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A electro guitar layered with glassy in the subway tunnel.",
        "Equalizer": [
            {"frequency": 100, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 12, "q": 1.8, "filter_type": "bell"},
            {"frequency": 15000, "gain": 10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A untuned performance of organ in a desert",
        "Equalizer": [
            {"frequency": 40, "gain": 8, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": -1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "A male speaks in an arena",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone treated with cloudy effect as if in a swamp",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": -3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": -5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": -8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.07,
            "diffusion": 0.9,
            "damping": 0.8,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Background audio with cloudy tones set in a library",
        "Equalizer": [
            {"frequency": 60, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": -4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.04,
            "diffusion": 0.6,
            "damping": 0.9,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone treated with saturated effect as if in a ship deck",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.05,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the bamboo grove with a chorused sound.",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 700, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2500, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 6000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the organ sound bitcrushed and reverberant in the valley.",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 8, "q": 2.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 4000, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 9000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.85,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the saxophone sound dry and reverberant in the living room.",
        "Equalizer": [
            {"frequency": 90, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.35,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient wet electro guitar heard in a radio studio.",
        "Equalizer": [
            {"frequency": 100, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live organ performance featuring radio-like effects in a busy intersection.",
        "Equalizer": [
            {"frequency": 100, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": 10, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 7000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.07,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A smoky female talks inside a museum",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A hangar filled with lightly glassy shimmer sounds",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.2,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient acoustic violin heard in a server room",
        "Equalizer": [
            {"frequency": 90, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of bass guitar made acoustic inside an arena",
        "Equalizer": [
            {"frequency": 50, "gain": 9, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 600, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "gated ambience fills the kitchen",
        "Equalizer": [
            {"frequency": 70, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.04,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live violin performance featuring cinematic effects in a small room.",
        "Equalizer": [
            {"frequency": 120, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4500, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live organ performance featuring radio-like effects in a garage.",
        "Equalizer": [
            {"frequency": 100, "gain": -10, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 9, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.07,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "monster-like texture applied to violin in a ship deck.",
        "Equalizer": [
            {"frequency": 50, "gain": 10, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": -6, "q": 2.0, "filter_type": "notch"},
            {"frequency": 3000, "gain": 9, "q": 1.8, "filter_type": "bell"},
            {"frequency": 7000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 25,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -7,
        }
    },
    {
        "prompt": "acoustic guitar played in the concert hall with a distorted sound.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of electro guitar with sparkling processing in a bamboo grove.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 12, "q": 1.6, "filter_type": "bell"},
            {"frequency": 15000, "gain": 9, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the piano sound tremolo and reverberant in the wine cellar",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the saxophone sound glowing and reverberant in the restaurant",
        "Equalizer": [
            {"frequency": 100, "gain": 1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "underwater texture applied to electro guitar in a bedroom",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": -4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": -8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": -12, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.9,
            "damping": 0.9,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "A drum set layered with stuttering in the forest",
        "Equalizer": [
            {"frequency": 70, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "trumpet played in the office cubicle with a reverse reverb sound",
        "Equalizer": [
            {"frequency": 90, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1,
            "pre_delay": 0.18,
            "diffusion": 0.3,
            "damping": 0.8,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "drum set treated with noisy effect as if in a bus station.",
        "Equalizer": [
            {"frequency": 70, "gain": 7, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.5,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male speaks in a recording studio.",
        "Equalizer": [
            {"frequency": 80, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.04,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of violin with alien processing in a server room.",
        "Equalizer": [
            {"frequency": 150, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -8, "q": 2.5, "filter_type": "notch"},
            {"frequency": 1800, "gain": 10, "q": 3.0, "filter_type": "bell"},
            {"frequency": 4500, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 9000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.8,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "A ambient performance of piano in a restaurant.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.5,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of organ made muffled inside a subway tunnel.",
        "Equalizer": [
            {"frequency": 60, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": -8, "q": 1.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -15, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.7,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Transform the speech of a male to sound distorted in a radio studio",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the female sound alien in the dining hall",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 2.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.07,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "Sound of piano made frosty inside a canyon",
        "Equalizer": [
            {"frequency": 60, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Sound ",
        "Equalizer": [
            {"frequency": 80, "gain": 1.0, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2.0, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3.0, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2.0, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2.0, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.17,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10.0,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0.0,
        }
    },
    {
        "prompt": "A live trumpet performance featuring broken effects in a hangar",
        "Equalizer": [
            {"frequency": 90, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.19,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "dry texture applied to trumpet in a cave.",
        "Equalizer": [
            {"frequency": 90, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.03,
            "diffusion": 0.2,
            "damping": 0.7,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient glowing bass guitar heard in a factory floor.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient tape-saturated acoustic guitar heard in a forest.",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 9000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.75,
            "damping": 0.45,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with a distorted character in the underground tunnel.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 11000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient ambient acoustic guitar heard in a lighthouse.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 9000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.14,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live saxophone performance featuring reverse reverb effects in an observatory dome",
        "Equalizer": [
            {"frequency": 90, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a female with an airy character in the cliff edge",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A whispering performance of electro guitar in a hangar",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.19,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute treated with cold effect as if in a shopping mall",
        "Equalizer": [
            {"frequency": 90, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4500, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "A organ layered with ring-modulated in the aquarium tunnel",
        "Equalizer": [
            {"frequency": 40, "gain": 8, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": -3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 2.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 5,
        }
    },
    {
        "prompt": "untuned texture applied to drum set in a closet.",
        "Equalizer": [
            {"frequency": 70, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.01,
            "diffusion": 0.2,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -0.4,
        }
    },
    {
        "prompt": "broken texture applied to electro guitar in a subway tunnel.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": -6, "q": 2.0, "filter_type": "notch"},
            {"frequency": 4500, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 22,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the hangar with a vibrato sound.",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 6000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "drum set treated with gated effect as if in a wind tunnel.",
        "Equalizer": [
            {"frequency": 70, "gain": 7, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Background audio with echoing tones set in a bedroom.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 1, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.04,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A violin layered with ghostly in the recording studio",
        "Equalizer": [
            {"frequency": 80, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 6, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Sound of keyboard made whispering inside a large hall",
        "Equalizer": [
            {"frequency": 70, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": -2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.9,
            "damping": 0.6,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with distorted processing in an open field",
        "Equalizer": [
            {"frequency": 90, "gain": 3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.13,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A drum set layered with ring-modulated in the large hall",
        "Equalizer": [
            {"frequency": 60, "gain": 7, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 14,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 6,
        }
    },
    {
        "prompt": "An ambient smeared saxophone heard in a cave",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": -1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.9,
            "damping": 0.7,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of saxophone with dry processing in a subway platform.",
        "Equalizer": [
            {"frequency": 90, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.8,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient frosty flute heard in a server room.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "acoustic guitar played in the wind tunnel with a compressed sound.",
        "Equalizer": [
            {"frequency": 120, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone played in the train station with a muffled sound.",
        "Equalizer": [
            {"frequency": 90, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3500, "gain": -8, "q": 1.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the bass guitar sound frosty and reverberant in the empty room.",
        "Equalizer": [
            {"frequency": 50, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.0,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0.5,
        }
    },
    {
        "prompt": "keyboard treated with telephone effect as if in an office cubicle",
        "Equalizer": [
            {"frequency": 80, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1,
            "pre_delay": 0.02,
            "diffusion": 0.4,
            "damping": 0.8,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A flute layered with stretched in the shopping mall",
        "Equalizer": [
            {"frequency": 90, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.11,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "flute played in the restaurant with an overdriven sound",
        "Equalizer": [
            {"frequency": 100, "gain": 1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the piano sound wobbly and reverberant in the train yard",
        "Equalizer": [
            {"frequency": 70, "gain": 4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "Make the keyboard sound vibrato and reverberant in the mountain peak",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.2,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A noisy performance of drum set in a beach.",
        "Equalizer": [
            {"frequency": 70, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 9, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A octave-up performance of organ in a beach.",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 12,
        }
    },
    {
        "prompt": "arena ambience with wet female.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound glassy and reverberant in the living room.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 12, "q": 1.6, "filter_type": "bell"},
            {"frequency": 15000, "gain": 9, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "lo-fi texture applied to acoustic guitar in a swamp.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -12, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.0,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with a buzzy character in the church",
        "Equalizer": [
            {"frequency": 90, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.13,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "piano treated with noisy effect as if in a courtyard",
        "Equalizer": [
            {"frequency": 70, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Transform the speech of a male to sound detuned in an elevator shaft",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.08,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.4,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "violin played in the open field with an untuned sound",
        "Equalizer": [
            {"frequency": 90, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 1, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -4,
        }
    },
    {
        "prompt": "Sound of drum set made octave-down inside a bedroom",
        "Equalizer": [
            {"frequency": 40, "gain": 10, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 500, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 1, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "cinematic rise texture applied to acoustic guitar in an auditorium.",
        "Equalizer": [
            {"frequency": 100, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4500, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 9, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin played in the auditorium with a compressed air sound.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A acoustic guitar layered with vibrato in the wind tunnel.",
        "Equalizer": [
            {"frequency": 120, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1800, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.5,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male whispers in the subway platform.",
        "Equalizer": [
            {"frequency": 80, "gain": -8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.5,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female whispers in the attic.",
        "Equalizer": [
            {"frequency": 80, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.5,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.8,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male is singing in a radio studio, and the voice is cinematic rise.",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2.0,
            "pre_delay": 0.04,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a cliff edge with echoing effects.",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "piano played in the oil refinery with a psychedelic sound.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -4, "q": 2.0, "filter_type": "notch"},
            {"frequency": 4000, "gain": 9, "q": 1.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 7, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female whispers in the conference center.",
        "Equalizer": [
            {"frequency": 80, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4500, "gain": 2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6.0,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.8,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A distant performance of drum set in a warehouse.",
        "Equalizer": [
            {"frequency": 70, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -10, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9.0,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A alien performance of saxophone in a concert hall",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": -3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 2.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 5,
        }
    },
    {
        "prompt": "Make the violin sound vintage and reverberant in the forest",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": -3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with a glitchy character in the beach",
        "Equalizer": [
            {"frequency": 90, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "plucky texture applied to acoustic guitar in an open field",
        "Equalizer": [
            {"frequency": 90, "gain": 4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.11,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female speaks in a stadium",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient metallic acoustic guitar heard in a concert hall",
        "Equalizer": [
            {"frequency": 90, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 7, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of bass guitar with alien processing in a church",
        "Equalizer": [
            {"frequency": 50, "gain": 8, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": -4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "A flute layered with compressed in the open field",
        "Equalizer": [
            {"frequency": 100, "gain": -1, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.11,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "muted texture applied to organ in a dining hall",
        "Equalizer": [
            {"frequency": 40, "gain": 6, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": -4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.07,
            "diffusion": 0.7,
            "damping": 0.7,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with octave-down processing in a museum",
        "Equalizer": [
            {"frequency": 40, "gain": 12, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 8, "q": 1.0, "filter_type": "bell"},
            {"frequency": 600, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 1, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "A drum set layered with metallic in the airport terminal.",
        "Equalizer": [
            {"frequency": 70, "gain": 7, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 9, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.13,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A frosty performance of electro guitar in a laboratory.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 6000, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.3,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "saxophone treated with percussive effect as if in a server room.",
        "Equalizer": [
            {"frequency": 90, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4500, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound ",
        "Equalizer": [
            {"frequency": 100, "gain": -12.0, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": -4.0, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6.0, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 15.0, "q": 2.0, "filter_type": "bell"},
            {"frequency": 15000, "gain": 12.0, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.1,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 0.0,
            "color": 0.0,
        },
        "Pitch": {
            "scale": 0.0,
        }
    },
    {
        "prompt": "A live drum set performance featuring bitcrushed effects in a stadium.",
        "Equalizer": [
            {"frequency": 70, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": 9, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live drum set performance featuring cinematic rise effects in a cliff edge",
        "Equalizer": [
            {"frequency": 60, "gain": 7, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Recording of organ with noisy processing in an amphitheater",
        "Equalizer": [
            {"frequency": 40, "gain": 8, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient crystal clear acoustic guitar heard in a stairwell",
        "Equalizer": [
            {"frequency": 80, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "trumpet treated with octave-down effect as if in a dining hall",
        "Equalizer": [
            {"frequency": 40, "gain": 10, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.0, "filter_type": "bell"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "Recording of organ with stretched processing in a theater",
        "Equalizer": [
            {"frequency": 40, "gain": 7, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.5, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.13,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "Sound of trumpet made robotic inside a concert hall.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of keyboard with squelchy processing in a stairwell.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -6, "q": 2.0, "filter_type": "notch"},
            {"frequency": 1500, "gain": 8, "q": 2.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 9, "q": 1.9, "filter_type": "bell"},
            {"frequency": 10000, "gain": -4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4.0,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.9,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live electro guitar performance featuring cinematic effects in a bamboo grove.",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female is singing in a pub, and the voice is plucky.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 9000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3.0,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something reversed in the anechoic chamber.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.1,
            "pre_delay": 0.01,
            "diffusion": 0.1,
            "damping": 1.0,
            "wet_gain": 0.05,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute treated with glassy shimmer effect as if in an underwater space",
        "Equalizer": [
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": -4, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.9,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient telephone keyboard heard in a mountain peak",
        "Equalizer": [
            {"frequency": 80, "gain": -10, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2500, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "detuned texture applied to piano in a factory floor",
        "Equalizer": [
            {"frequency": 70, "gain": 4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 1, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.4,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "Sound of keyboard made chorused inside a ballroom",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A keyboard layered with metallic in the ship deck",
        "Equalizer": [
            {"frequency": 90, "gain": 3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "organ treated with percussive effect as if in an open field.",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10.0,
            "pre_delay": 0.2,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a library with untuned effects.",
        "Equalizer": [
            {"frequency": 80, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": -2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 4000, "gain": -4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.8,
            "pre_delay": 0.03,
            "diffusion": 0.3,
            "damping": 0.9,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -0.3,
        }
    },
    {
        "prompt": "Recording of saxophone with broken processing in a forest.",
        "Equalizer": [
            {"frequency": 90, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 350, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": -5, "q": 2.0, "filter_type": "notch"},
            {"frequency": 3500, "gain": 9, "q": 1.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.5,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 22,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "closet ambience with alien male.",
        "Equalizer": [
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -6, "q": 2.5, "filter_type": "notch"},
            {"frequency": 1800, "gain": 10, "q": 3.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 8, "q": 1.8, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.8,
            "pre_delay": 0.02,
            "diffusion": 0.3,
            "damping": 0.8,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": -4,
        }
    },
    {
        "prompt": "pitch-shifted down texture applied to flute in an office cubicle.",
        "Equalizer": [
            {"frequency": 60, "gain": 8, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 3000, "gain": -2, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.0, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1.5,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": -8,
        }
    },
    {
        "prompt": "flute played in the underwater space with a pulsing sound.",
        "Equalizer": [
            {"frequency": 150, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 16000, "gain": -4, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A glassy shimmer performance of flute in a subway tunnel.",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.5, "filter_type": "bell"},
            {"frequency": 16000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.05,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something vibrato in the ballroom.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with an alien character in the bedroom.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.9, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 2.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": -3, "q": 1.5, "filter_type": "notch"},
            {"frequency": 9000, "gain": 4, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.05,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.7,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "keyboard treated with compressed air effect as if in an observatory dome.",
        "Equalizer": [
            {"frequency": 120, "gain": -5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 14000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.75,
            "damping": 0.3,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of saxophone with breathy processing in a museum",
        "Equalizer": [
            {"frequency": 20000, "gain": -5, "q": 1.2, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 250, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 0.8, "filter_type": "notch"},
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "piano treated with cold effect as if in an auditorium",
        "Equalizer": [
            {"frequency": 18000, "gain": 6, "q": 1.3, "filter_type": "high-shelf"},
            {"frequency": 4000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 600, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 150, "gain": -1, "q": 1.5, "filter_type": "low-shelf"},
            {"frequency": 250, "gain": -4, "q": 0.7, "filter_type": "notch"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A trumpet layered with crystal clear in the valley",
        "Equalizer": [
            {"frequency": 16000, "gain": 5, "q": 1.1, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 200, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 50, "gain": -3, "q": 0.8, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.45,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a parking garage with alien effects",
        "Equalizer": [
            {"frequency": 180, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1000, "gain": -5, "q": 0.7, "filter_type": "notch"},
            {"frequency": 4000, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 300, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "Make the female sound lo-fi in the cave",
        "Equalizer": [
            {"frequency": 4000, "gain": -7, "q": 1.0, "filter_type": "bell"},
            {"frequency": 150, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 1000, "gain": -4, "q": 1.5, "filter_type": "notch"},
            {"frequency": 6000, "gain": -10, "q": 1.3, "filter_type": "bell"},
            {"frequency": 80, "gain": 4, "q": 0.9, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.06,
            "diffusion": 0.7,
            "damping": 0.9,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.5,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound untuned and reverberant in the anechoic chamber.",
        "Equalizer": [
            {"frequency": 80, "gain": -5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 700, "gain": -6, "q": 2.0, "filter_type": "notch"},
            {"frequency": 2000, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0,
            "pre_delay": 0,
            "diffusion": 0,
            "damping": 0,
            "wet_gain": 1,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "A live electro guitar performance featuring dry effects in a bedroom.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 14000, "gain": -1, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0,
            "diffusion": 0.2,
            "damping": 0.6,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live electro guitar performance featuring glassy effects in an open-air market.",
        "Equalizer": [
            {"frequency": 150, "gain": -2, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 16000, "gain": 6, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of drum set with glassy processing in a bar.",
        "Equalizer": [
            {"frequency": 60, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -4, "q": 1.7, "filter_type": "notch"},
            {"frequency": 1500, "gain": 5, "q": 2, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A robotic performance of organ in a bathroom.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.75,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "electro guitar played in the bamboo grove with a percussive sound",
        "Equalizer": [
            {"frequency": 3000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 150, "gain": -2, "q": 0.8, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 1.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.04,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.35,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "glassy shimmer texture applied to keyboard in a stadium",
        "Equalizer": [
            {"frequency": 8000, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.0, "filter_type": "high-shelf"},
            {"frequency": 2000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 400, "gain": -3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 15000, "gain": 8, "q": 1.5, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the electro guitar sound breathy and reverberant in the nightclub",
        "Equalizer": [
            {"frequency": 4000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 7000, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 250, "gain": 3, "q": 0.9, "filter_type": "low-shelf"},
            {"frequency": 6000, "gain": 3, "q": 1.5, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.55,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of bass guitar with spacey processing in a forest",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 500, "gain": -2, "q": 0.8, "filter_type": "bell"},
            {"frequency": 5000, "gain": 4, "q": 1.5, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.8,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "organ treated with granular effect as if in a cave",
        "Equalizer": [
            {"frequency": 120, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 400, "gain": -3, "q": 0.9, "filter_type": "notch"},
            {"frequency": 6000, "gain": 2, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.9,
            "wet_gain": 0.65,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.7,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound glowing and reverberant in the stairwell.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Background audio with alien tones set in a church.",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -3, "q": 2.0, "filter_type": "notch"},
            {"frequency": 1500, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.7,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "A female whispers in the cathedral.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 3, "q": 1.9, "filter_type": "bell"},
            {"frequency": 2000, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live keyboard performance featuring metallic effects in a bedroom.",
        "Equalizer": [
            {"frequency": 150, "gain": -4, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 3, "q": 1.7, "filter_type": "bell"},
            {"frequency": 2000, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 15000, "gain": 6, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of trumpet made pitch-shifted up inside an elevator shaft.",
        "Equalizer": [
            {"frequency": 120, "gain": -5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a gymnasium with overdriven effects",
        "Equalizer": [
            {"frequency": 200, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": -3, "q": 0.8, "filter_type": "notch"},
            {"frequency": 3000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 80, "gain": 2, "q": 1.1, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Environmental sound with acoustic characteristics in an underground tunnel",
        "Equalizer": [
            {"frequency": 150, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 300, "gain": -2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 8000, "gain": -4, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "percussive texture applied to drum set in a silo",
        "Equalizer": [
            {"frequency": 2500, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 400, "gain": -3, "q": 0.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something detuned in the basement",
        "Equalizer": [
            {"frequency": 2000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 200, "gain": 1, "q": 0.9, "filter_type": "low-shelf"},
            {"frequency": 6000, "gain": 1, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.8,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "Make the male sound dry in the nightclub",
        "Equalizer": [
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 200, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 3000, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 500, "gain": -2, "q": 0.8, "filter_type": "bell"},
            {"frequency": 5000, "gain": 1, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.02,
            "diffusion": 0.3,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute played in the oil refinery with a noisy sound.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 3, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 2.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A drum set layered with vibrato in the garage.",
        "Equalizer": [
            {"frequency": 60, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -2, "q": 1.7, "filter_type": "notch"},
            {"frequency": 1200, "gain": 6, "q": 1.9, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A alien soundscape in an empty room.",
        "Equalizer": [
            {"frequency": 80, "gain": -6, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": -4, "q": 2.0, "filter_type": "notch"},
            {"frequency": 1800, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4500, "gain": 8, "q": 1.3, "filter_type": "bell"},
            {"frequency": 15000, "gain": 5, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -7,
        }
    },
    {
        "prompt": "A male is shouting in a closet with a hollow tone.",
        "Equalizer": [
            {"frequency": 120, "gain": -5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -6, "q": 2.2, "filter_type": "notch"},
            {"frequency": 1200, "gain": 8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3000, "gain": 6, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -4, "q": 0.8, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1,
            "pre_delay": 0.02,
            "diffusion": 0.3,
            "damping": 0.8,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female is singing in an attic, and the voice is psychedelic.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.9, "filter_type": "bell"},
            {"frequency": 2000, "gain": -3, "q": 1.8, "filter_type": "notch"},
            {"frequency": 5000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 3,
        }
    },
    {
        "prompt": "An ambient ambient saxophone heard in an airplane cabin",
        "Equalizer": [
            {"frequency": 3000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 500, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 150, "gain": 1, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 1200, "gain": -1, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.02,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.25,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Pure octave-down effect reminiscent of a parking garage",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 1500, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": -4, "q": 1.2, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": -12,
        }
    },
    {
        "prompt": "Sound of acoustic guitar made wet inside a silo",
        "Equalizer": [
            {"frequency": 2000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 400, "gain": -2, "q": 0.8, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "nostalgic texture applied to electro guitar in a bathroom",
        "Equalizer": [
            {"frequency": 1000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 6000, "gain": -4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 300, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 4000, "gain": 1, "q": 1.2, "filter_type": "bell"},
            {"frequency": 150, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of keyboard with fuzzed processing in a subway tunnel",
        "Equalizer": [
            {"frequency": 2500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 200, "gain": 2, "q": 0.9, "filter_type": "low-shelf"},
            {"frequency": 1200, "gain": -2, "q": 0.8, "filter_type": "notch"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.15,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 22,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "tuned texture applied to organ in an airport terminal.",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bitcrushed texture applied to organ in a lighthouse.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.9, "filter_type": "bell"},
            {"frequency": 2000, "gain": -3, "q": 2.0, "filter_type": "notch"},
            {"frequency": 5000, "gain": 8, "q": 1.5, "filter_type": "bell"},
            {"frequency": 12000, "gain": -5, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with vibrato processing in a closet.",
        "Equalizer": [
            {"frequency": 120, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.02,
            "diffusion": 0.3,
            "damping": 0.8,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a pub with crystal clear effects.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "keyboard treated with ambient effect as if in an attic.",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 14000, "gain": 5, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with a metallic character in the observatory dome",
        "Equalizer": [
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 300, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "hazy texture applied to electro guitar in an empty room",
        "Equalizer": [
            {"frequency": 2000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 6000, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 10000, "gain": -5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 400, "gain": 1, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.06,
            "diffusion": 0.4,
            "damping": 0.8,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live piano performance featuring echoing effects in a power plant",
        "Equalizer": [
            {"frequency": 2500, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1200, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male speaks in a swamp",
        "Equalizer": [
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 300, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 3000, "gain": 1, "q": 1.1, "filter_type": "bell"},
            {"frequency": 6000, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 150, "gain": 1, "q": 1.0, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.04,
            "diffusion": 0.3,
            "damping": 0.9,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A crystal clear performance of keyboard in a server room",
        "Equalizer": [
            {"frequency": 3000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 500, "gain": 1, "q": 0.9, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.4, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.02,
            "diffusion": 0.4,
            "damping": 0.6,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the opera house with a compressed sound.",
        "Equalizer": [
            {"frequency": 60, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 800, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2500, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": -2, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "trumpet treated with tremolo effect as if in a wine cellar.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.06,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "trumpet treated with sidechained effect as if in a ship deck.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.5,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A buzzy performance of organ in an open field.",
        "Equalizer": [
            {"frequency": 80, "gain": -2, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 6, "q": 1.9, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3500, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.02,
            "diffusion": 0.2,
            "damping": 0.1,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A bass guitar layered with sidechained in the train station.",
        "Equalizer": [
            {"frequency": 50, "gain": 4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 600, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2000, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -1, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of keyboard made robotic inside a train yard",
        "Equalizer": [
            {"frequency": 2500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": -2, "q": 0.8, "filter_type": "notch"},
            {"frequency": 8000, "gain": 5, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "Make the flute sound hazy and reverberant in the wine cellar",
        "Equalizer": [
            {"frequency": 3000, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 6000, "gain": -4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.1, "filter_type": "bell"},
            {"frequency": 400, "gain": 1, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.8,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of saxophone with glowing processing in a rocky shore",
        "Equalizer": [
            {"frequency": 4000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 10000, "gain": 4, "q": 1.3, "filter_type": "high-shelf"},
            {"frequency": 300, "gain": 2, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "trumpet treated with distant effect as if in a power plant",
        "Equalizer": [
            {"frequency": 2000, "gain": -2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 1, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": -5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 400, "gain": 2, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of drum set made plucky inside a cafeteria",
        "Equalizer": [
            {"frequency": 3000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 200, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Recording of flute with detuned processing in an empty room.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.5,
        },
        "Pitch": {
            "scale": -2,
        }
    },
    {
        "prompt": "saxophone treated with chorused effect as if in a beach.",
        "Equalizer": [
            {"frequency": 120, "gain": -2, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1200, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "cold texture applied to organ in a recording studio.",
        "Equalizer": [
            {"frequency": 80, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -3, "q": 1.8, "filter_type": "notch"},
            {"frequency": 1500, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A electro guitar layered with ring-modulated in the courtyard.",
        "Equalizer": [
            {"frequency": 150, "gain": -3, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1800, "gain": -4, "q": 2.0, "filter_type": "notch"},
            {"frequency": 3500, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 9000, "gain": 5, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live electro guitar performance featuring breathy effects in an arena.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 15000, "gain": 6, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of bass guitar made saturated inside a lighthouse",
        "Equalizer": [
            {"frequency": 80, "gain": 6, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2000, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.09,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.45,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with pulsing processing in a conference room",
        "Equalizer": [
            {"frequency": 2000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 100, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 1, "q": 0.9, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.04,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.35,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with an echoing character in the large hall",
        "Equalizer": [
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 300, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 2000, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 5000, "gain": 3, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A tape-saturated performance of violin in a rocky shore",
        "Equalizer": [
            {"frequency": 3000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 500, "gain": 1, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A saxophone layered with stretched in the church",
        "Equalizer": [
            {"frequency": 2500, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 1, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": 3, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A drum set layered with smoky in the church.",
        "Equalizer": [
            {"frequency": 60, "gain": 5, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": -2, "q": 1.8, "filter_type": "notch"},
            {"frequency": 1000, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A recording of a male with a sidechained character in the rocky shore.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.07,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of saxophone made plucky inside a concert hall.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1200, "gain": 7, "q": 1.9, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.9,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the bass guitar sound warm and reverberant in the power plant.",
        "Equalizer": [
            {"frequency": 50, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2500, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "kitchen ambience with crystal clear male.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.4,
            "damping": 0.7,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the male sound compressed air in the rocky shore",
        "Equalizer": [
            {"frequency": 1000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 300, "gain": 1, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin played in the library with a ring-modulated sound",
        "Equalizer": [
            {"frequency": 2500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 3500, "gain": 3, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.8,
            "wet_gain": 0.35,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "A organ layered with expanded in the subway platform",
        "Equalizer": [
            {"frequency": 120, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 5000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 400, "gain": -2, "q": 0.9, "filter_type": "notch"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the organ sound spacey and reverberant in the canyon",
        "Equalizer": [
            {"frequency": 150, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 1000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 500, "gain": -1, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Environmental sound with lo-fi characteristics in an opera house",
        "Equalizer": [
            {"frequency": 4000, "gain": -6, "q": 1.1, "filter_type": "bell"},
            {"frequency": 8000, "gain": -8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 300, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 150, "gain": 2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 6000, "gain": -5, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.14,
            "diffusion": 0.8,
            "damping": 0.9,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 14,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "drum set treated with wet effect as if in an underground tunnel.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": -3, "q": 1.8, "filter_type": "notch"},
            {"frequency": 1200, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.6,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A electro guitar layered with acoustic in the church.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "cathedral ambience with wet male.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.18,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live organ performance featuring muffled effects in a church.",
        "Equalizer": [
            {"frequency": 80, "gain": 3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": -4, "q": 1.4, "filter_type": "notch"},
            {"frequency": 8000, "gain": -6, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.14,
            "diffusion": 0.8,
            "damping": 0.7,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female is shouting in a cathedral with a sidechained tone.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2000, "gain": 8, "q": 1.7, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.17,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 9,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Pure sparkling effect reminiscent of a stairwell",
        "Equalizer": [
            {"frequency": 8000, "gain": 8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.1, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 15000, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 2000, "gain": 2, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the electro guitar sound buzzy and reverberant in the laboratory",
        "Equalizer": [
            {"frequency": 2500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 500, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": 3, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.55,
        },
        "Distortion": {
            "gain": 16,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female speaks in a stairwell",
        "Equalizer": [
            {"frequency": 2000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 5000, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 400, "gain": 1, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.07,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin treated with sparkling effect as if in a desert",
        "Equalizer": [
            {"frequency": 3000, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 8000, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.3, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 5, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.08,
            "diffusion": 0.3,
            "damping": 0.5,
            "wet_gain": 0.35,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female speaks in a stadium",
        "Equalizer": [
            {"frequency": 2000, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 500, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 5000, "gain": 2, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "vibrato texture applied to flute in a factory floor.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "Simulate the acoustic environment of a radio studio with vintage effects.",
        "Equalizer": [
            {"frequency": 80, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": -4, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.05,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of acoustic guitar made cinematic inside a bathroom.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1200, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.08,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of keyboard with robotic processing in a stairwell.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.9, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "ballroom ambience with cinematic female.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2000, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "The aura of crystal clear in the nightclub",
        "Equalizer": [
            {"frequency": 8000, "gain": 8, "q": 1.2, "filter_type": "bell"},
            {"frequency": 12000, "gain": 7, "q": 1.1, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 2000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 15000, "gain": 8, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.4,
            "wet_gain": 0.45,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the drum set sound dry and reverberant in the airport terminal",
        "Equalizer": [
            {"frequency": 2500, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 4000, "gain": 3, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.8,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the acoustic guitar sound fuzzed and reverberant in the conference center",
        "Equalizer": [
            {"frequency": 2000, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": -2, "q": 0.8, "filter_type": "notch"},
            {"frequency": 6000, "gain": 6, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.09,
            "diffusion": 0.7,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A gated performance of violin in a classroom",
        "Equalizer": [
            {"frequency": 3000, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.04,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin treated with broken effect as if in a busy intersection",
        "Equalizer": [
            {"frequency": 2500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": -3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": 4, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.06,
            "diffusion": 0.4,
            "damping": 0.5,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "A underwater male talks inside a canyon.",
        "Equalizer": [
            {"frequency": 100, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3000, "gain": -4, "q": 1.4, "filter_type": "notch"},
            {"frequency": 8000, "gain": -8, "q": 0.6, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": -1,
        }
    },
    {
        "prompt": "Sound of violin made frosty inside a subway platform.",
        "Equalizer": [
            {"frequency": 120, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -3, "q": 1.9, "filter_type": "notch"},
            {"frequency": 1500, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 7, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female is shouting in a city street with an untuned tone.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2000, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": -3,
        }
    },
    {
        "prompt": "fuzzed texture applied to keyboard in an attic.",
        "Equalizer": [
            {"frequency": 100, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 700, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 8, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "flute treated with ambient effect as if in a concert hall.",
        "Equalizer": [
            {"frequency": 80, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 4, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.9,
            "damping": 0.2,
            "wet_gain": 0.9,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "violin played in the cave with a pitch-shifted up sound",
        "Equalizer": [
            {"frequency": 3000, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.7,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.1,
        },
        "Pitch": {
            "scale": 4,
        }
    },
    {
        "prompt": "Make the drum set sound granular and reverberant in the oil refinery",
        "Equalizer": [
            {"frequency": 2500, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 18,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "drum set played in the beach with an acoustic sound",
        "Equalizer": [
            {"frequency": 2500, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 100, "gain": 3, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 5000, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 0.9, "filter_type": "bell"},
            {"frequency": 8000, "gain": 1, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.05,
            "diffusion": 0.4,
            "damping": 0.6,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live bass guitar performance featuring tape-saturated effects in a gymnasium",
        "Equalizer": [
            {"frequency": 80, "gain": 5, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 3000, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": -3, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.1,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 14,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A hazy performance of bass guitar in a theater",
        "Equalizer": [
            {"frequency": 100, "gain": 4, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1500, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": -4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": -6, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.08,
            "diffusion": 0.7,
            "damping": 0.8,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "acoustic guitar played in the train station with a percussive sound.",
        "Equalizer": [
            {"frequency": 80, "gain": -3, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 300, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1200, "gain": 7, "q": 1.9, "filter_type": "bell"},
            {"frequency": 3500, "gain": 6, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.12,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A flute layered with sidechained in the closet.",
        "Equalizer": [
            {"frequency": 100, "gain": -4, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 7, "q": 1.7, "filter_type": "bell"},
            {"frequency": 3500, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 0.5,
            "pre_delay": 0.02,
            "diffusion": 0.3,
            "damping": 0.8,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "saxophone played in the gymnasium with a noisy sound.",
        "Equalizer": [
            {"frequency": 120, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 6, "q": 1.6, "filter_type": "bell"},
            {"frequency": 1200, "gain": 7, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -1, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.11,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A live bass guitar performance featuring monster-like effects in an aquarium tunnel.",
        "Equalizer": [
            {"frequency": 40, "gain": 8, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 600, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2000, "gain": -4, "q": 1.8, "filter_type": "notch"},
            {"frequency": 5000, "gain": 7, "q": 1.3, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.09,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 20,
            "color": 0.9,
        },
        "Pitch": {
            "scale": -5,
        }
    },
    {
        "prompt": "An ambient muffled acoustic guitar heard in an auditorium.",
        "Equalizer": [
            {"frequency": 120, "gain": -2, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4, "q": 1.8, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 4000, "gain": -5, "q": 1.4, "filter_type": "notch"},
            {"frequency": 8000, "gain": -7, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.15,
            "diffusion": 0.9,
            "damping": 0.6,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the cafeteria with an octave-up sound",
        "Equalizer": [
            {"frequency": 160, "gain": 5, "q": 1.0, "filter_type": "bell"},
            {"frequency": 400, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1000, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2500, "gain": 2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 80, "gain": -2, "q": 0.9, "filter_type": "low-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 12,
        }
    },
    {
        "prompt": "keyboard played in the living room with a noisy sound",
        "Equalizer": [
            {"frequency": 2000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 5000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 8000, "gain": 5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 300, "gain": 1, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.3,
        },
        "Distortion": {
            "gain": 12,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A fuzzed performance of electro guitar in an oil refinery",
        "Equalizer": [
            {"frequency": 2500, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 500, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": 5, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.14,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 22,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A flute layered with stretched in the power plant",
        "Equalizer": [
            {"frequency": 3000, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1500, "gain": 2, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 3, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1000, "gain": 1, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 10,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A plucky performance of flute in a lighthouse",
        "Equalizer": [
            {"frequency": 3000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4, "q": 1.1, "filter_type": "bell"},
            {"frequency": 1000, "gain": 2, "q": 0.9, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7,
            "pre_delay": 0.09,
            "diffusion": 0.6,
            "damping": 0.5,
            "wet_gain": 0.45,
        },
        "Distortion": {
            "gain": 6,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of bass guitar made stretched inside a cliff edge.",
        "Equalizer": [
            {"frequency": 40, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 600, "gain": 3, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2000, "gain": -3, "q": 1.8, "filter_type": "notch"},
            {"frequency": 8000, "gain": 4, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.08,
            "diffusion": 0.5,
            "damping": 0.3,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": -4,
        }
    },
    {
        "prompt": "organ played in the ballroom with a tremolo sound.",
        "Equalizer": [
            {"frequency": 80, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1500, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3500, "gain": 4, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 2, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 1,
        }
    },
    {
        "prompt": "A live bass guitar performance featuring gated effects in a desert.",
        "Equalizer": [
            {"frequency": 50, "gain": 7, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2500, "gain": 5, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": 2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 1,
            "pre_delay": 0.02,
            "diffusion": 0.2,
            "damping": 0.9,
            "wet_gain": 0.1,
        },
        "Distortion": {
            "gain": 9,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient formant-shifted saxophone heard in a conference room.",
        "Equalizer": [
            {"frequency": 120, "gain": -3, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 600, "gain": -2, "q": 2.0, "filter_type": "notch"},
            {"frequency": 1200, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 3000, "gain": 7, "q": 1.4, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 4,
            "pre_delay": 0.06,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 4,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 2,
        }
    },
    {
        "prompt": "A female whispers in the ballroom.",
        "Equalizer": [
            {"frequency": 100, "gain": -6, "q": 1.4, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2500, "gain": 8, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.9,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A male whispers in the airport terminal",
        "Equalizer": [
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 300, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 3000, "gain": 5, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 500, "gain": 2, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.6,
            "damping": 0.6,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "organ played in the ship deck with a sidechained sound",
        "Equalizer": [
            {"frequency": 120, "gain": 5, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 2000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 4000, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 400, "gain": -2, "q": 0.9, "filter_type": "notch"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.06,
            "diffusion": 0.4,
            "damping": 0.5,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 10,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound of violin made crystal clear inside a planetarium",
        "Equalizer": [
            {"frequency": 3000, "gain": 5, "q": 1.1, "filter_type": "bell"},
            {"frequency": 8000, "gain": 7, "q": 1.2, "filter_type": "bell"},
            {"frequency": 1500, "gain": 3, "q": 1.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 6, "q": 1.3, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 4, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.4,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "An ambient sparkling drum set heard in a cliff edge",
        "Equalizer": [
            {"frequency": 3000, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 8000, "gain": 6, "q": 1.2, "filter_type": "bell"},
            {"frequency": 200, "gain": 1, "q": 1.0, "filter_type": "bell"},
            {"frequency": 12000, "gain": 5, "q": 1.3, "filter_type": "high-shelf"},
            {"frequency": 5000, "gain": 4, "q": 1.4, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.08,
            "diffusion": 0.3,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 2,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female says something smoky in the bathroom",
        "Equalizer": [
            {"frequency": 2000, "gain": 2, "q": 1.1, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 4000, "gain": -2, "q": 1.2, "filter_type": "bell"},
            {"frequency": 300, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 6000, "gain": -3, "q": 1.0, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 2,
            "pre_delay": 0.03,
            "diffusion": 0.5,
            "damping": 0.7,
            "wet_gain": 0.4,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.5,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A female whispers in the conference center.",
        "Equalizer": [
            {"frequency": 100, "gain": -5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 800, "gain": 5, "q": 1.8, "filter_type": "bell"},
            {"frequency": 2500, "gain": 7, "q": 1.6, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 12000, "gain": 4, "q": 0.5, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 6,
            "pre_delay": 0.1,
            "diffusion": 0.8,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of drum set with dry processing in a library.",
        "Equalizer": [
            {"frequency": 60, "gain": 4, "q": 1.2, "filter_type": "low-shelf"},
            {"frequency": 200, "gain": -2, "q": 1.7, "filter_type": "notch"},
            {"frequency": 1000, "gain": 6, "q": 1.8, "filter_type": "bell"},
            {"frequency": 4000, "gain": 5, "q": 1.4, "filter_type": "bell"},
            {"frequency": 10000, "gain": 3, "q": 0.6, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 3,
            "pre_delay": 0.04,
            "diffusion": 0.6,
            "damping": 0.8,
            "wet_gain": 0.2,
        },
        "Distortion": {
            "gain": 1,
            "color": 0.2,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Recording of acoustic guitar with telephone processing in a lighthouse.",
        "Equalizer": [
            {"frequency": 300, "gain": -8, "q": 1.5, "filter_type": "highpass"},
            {"frequency": 800, "gain": 6, "q": 2.0, "filter_type": "bell"},
            {"frequency": 1500, "gain": 5, "q": 1.9, "filter_type": "bell"},
            {"frequency": 3000, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 6000, "gain": -10, "q": 0.8, "filter_type": "lowpass"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.08,
            "diffusion": 0.6,
            "damping": 0.4,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 8,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Make the organ sound warm and reverberant in the wind tunnel.",
        "Equalizer": [
            {"frequency": 60, "gain": 6, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 5, "q": 1.7, "filter_type": "bell"},
            {"frequency": 1200, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3, "q": 1.4, "filter_type": "bell"},
            {"frequency": 8000, "gain": -2, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.13,
            "diffusion": 0.7,
            "damping": 0.3,
            "wet_gain": 0.8,
        },
        "Distortion": {
            "gain": 3,
            "color": 0.4,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "bass guitar played in the arena with a compressed sound.",
        "Equalizer": [
            {"frequency": 50, "gain": 5, "q": 1.3, "filter_type": "low-shelf"},
            {"frequency": 150, "gain": 6, "q": 1.7, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.6, "filter_type": "bell"},
            {"frequency": 2500, "gain": 3, "q": 1.5, "filter_type": "bell"},
            {"frequency": 8000, "gain": -1, "q": 0.7, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.16,
            "diffusion": 0.8,
            "damping": 0.3,
            "wet_gain": 0.7,
        },
        "Distortion": {
            "gain": 7,
            "color": 0.6,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A cloudy performance of electro guitar in a restaurant",
        "Equalizer": [
            {"frequency": 2000, "gain": -3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 5000, "gain": -5, "q": 1.3, "filter_type": "bell"},
            {"frequency": 800, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 150, "gain": 3, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 12000, "gain": -4, "q": 1.4, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 5,
            "pre_delay": 0.05,
            "diffusion": 0.6,
            "damping": 0.7,
            "wet_gain": 0.5,
        },
        "Distortion": {
            "gain": 5,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "A piano layered with alien in the shopping mall",
        "Equalizer": [
            {"frequency": 3000, "gain": 4, "q": 1.2, "filter_type": "bell"},
            {"frequency": 7000, "gain": 6, "q": 1.3, "filter_type": "bell"},
            {"frequency": 150, "gain": -2, "q": 1.0, "filter_type": "low-shelf"},
            {"frequency": 500, "gain": 3, "q": 1.1, "filter_type": "bell"},
            {"frequency": 10000, "gain": 5, "q": 1.4, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 15,
            "color": 0.8,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound ",
        "Equalizer": [
            {"frequency": 150, "gain": -5.0, "q": 1.1, "filter_type": "low-shelf"},
            {"frequency": 400, "gain": 4.0, "q": 1.0, "filter_type": "bell"},
            {"frequency": 1200, "gain": 5.0, "q": 1.2, "filter_type": "bell"},
            {"frequency": 3000, "gain": 3.0, "q": 1.3, "filter_type": "bell"},
            {"frequency": 8000, "gain": 4.0, "q": 1.1, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 7.0,
            "pre_delay": 0.08,
            "diffusion": 0.5,
            "damping": 0.6,
            "wet_gain": 0.55,
        },
        "Distortion": {
            "gain": 2.0,
            "color": 0.3,
        },
        "Pitch": {
            "scale": 12.0,
        }
    },
    {
        "prompt": "A female speaks in a hangar",
        "Equalizer": [
            {"frequency": 1000, "gain": 4, "q": 1.0, "filter_type": "bell"},
            {"frequency": 300, "gain": 3, "q": 0.9, "filter_type": "bell"},
            {"frequency": 2500, "gain": 3, "q": 1.2, "filter_type": "bell"},
            {"frequency": 6000, "gain": 2, "q": 1.3, "filter_type": "bell"},
            {"frequency": 400, "gain": 1, "q": 0.8, "filter_type": "bell"},
        ],
        "Reverb": {
            "room_size": 9,
            "pre_delay": 0.12,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 0,
            "color": 0,
        },
        "Pitch": {
            "scale": 0,
        }
    },
    {
        "prompt": "Sound ",
        "Equalizer": [
            {"frequency": 2500, "gain": 5.0, "q": 1.2, "filter_type": "bell"},
            {"frequency": 800, "gain": 3.0, "q": 1.0, "filter_type": "bell"},
            {"frequency": 5000, "gain": 6.0, "q": 1.3, "filter_type": "bell"},
            {"frequency": 400, "gain": -2.0, "q": 0.9, "filter_type": "notch"},
            {"frequency": 12000, "gain": 4.0, "q": 1.4, "filter_type": "high-shelf"},
        ],
        "Reverb": {
            "room_size": 8.0,
            "pre_delay": 0.1,
            "diffusion": 0.7,
            "damping": 0.5,
            "wet_gain": 0.6,
        },
        "Distortion": {
            "gain": 15.0,
            "color": 0.7,
        },
        "Pitch": {
            "scale": 0.0,
        }
    }
]
