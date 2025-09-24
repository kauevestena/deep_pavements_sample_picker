# Deep Pavements Sample Picker
Deep Pavements is a project to train a Convolutional Neural Network to do semantic segmentation.
This is the part where the samples got collected, to then be labeled. It leverages the capacities of the great [Language Segment-Anything
](https://github.com/luca-medeiros/lang-segment-anything) that's capable of predicting themes not present on training.

### Please refer to SETUP.md to start

## Usage

The sample picker can be configured using command line arguments:

### Command Line Arguments

- `--classes` or `-c`: List of class prompts for detection
- `--territories` or `-t`: List of territories with weights in format "territory:weight"

### Examples

**Basic usage with default values:**
```bash
python run.py
```
This uses default classes: `tree`, `vehicle` and default territories: `Vitorino Brazil:1`, `Curitiba Brazil:1`, `Milan Italy:1`, `Arcole Italy:1`

**Custom classes:**
```bash
python run.py --classes road sidewalk building tree
```

**Custom territories with weights:**
```bash
python run.py --territories "Paris France:2" "London UK:1" "Tokyo Japan:3"
```
Territory weights determine how often that territory is sampled. In the example above, Paris France will be sampled twice as often as London UK, and Tokyo Japan three times as often.

**Combined custom configuration:**
```bash
python run.py --classes road sidewalk crosswalk --territories "New York USA:2" "Los Angeles USA:1"
```

**Short form arguments:**
```bash
python run.py -c road sidewalk -t "Berlin Germany:1" "Munich Germany:2"
```

**Real-world example for pavement analysis:**
```bash
python run.py \
  --classes "road surface" "asphalt" "concrete" "paving stones" "crosswalk" "manhole cover" \
  --territories "San Francisco USA:3" "Amsterdam Netherlands:2" "Copenhagen Denmark:1"
```

**Urban infrastructure analysis:**
```bash
python run.py \
  -c "bike lane" "sidewalk" "curb" "street sign" "traffic light" \
  -t "Portland USA:2" "Barcelona Spain:2" "Vienna Austria:1"
```

### Legacy CSV Configuration

For backward compatibility, the application still supports CSV-based configuration. If no command line arguments are provided, it will fall back to reading from:
- `configs/prompted_classes.csv` - for class prompts
- `configs/territories.csv` - for territories and weights

However, **command line arguments take precedence** over CSV files when provided.

A frozen version of the working requirements was made at "frozen_requirements.txt", in case of image building failure with latest packages getting updated

#### Monitoring GPU usage:

execute gpustat:

    gpustat -cp --watch

or for different intervals, 3 seconds in the example:

    watch -n 3 -c gpustat -cp --color