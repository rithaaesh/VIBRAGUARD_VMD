# Dataset Adapter & Schema

## Primary Reference Dataset
- **CWRU Bearing Dataset**: Case Western Reserve University Bearing Data Center.
- **Sampling Frequencies**: 12,000 Hz / 48,000 Hz.
- **Fault Conditions**: Normal, Ball Fault, Inner Race Fault, Outer Race Fault.

## Generic CSV Upload Requirements
- Columns: `vibration` (float, required), `timestamp` (optional).
- Metadata: `sampling_rate` (Hz), `rpm` (RPM), `load_hp` (HP), `machine_id` (String). Unspecified fields are recorded as `Unknown / Not Provided`.
