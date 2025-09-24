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
- `--bbox` or `-b`: Bounding box in format "min_lon,min_lat,max_lon,max_lat" (alternative to territories)

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

**Using direct bounding box (San Francisco downtown):**
```bash
python run.py --bbox "-122.42,37.76,-122.38,37.80" --classes "road surface" "crosswalk" "traffic light"
```

**Using bbox for specific area analysis (Central Paris):**
```bash  
python run.py -b "2.32,48.85,2.37,48.87" -c "cobblestone" "asphalt" "sidewalk" "tree"
```

**Bbox coordinates for major cities:**
- San Francisco: `-122.5,37.7,-122.4,37.8`
- Paris: `2.2,48.8,2.4,48.9`
- London: `-0.2,51.4,0.1,51.6`
- New York: `-74.1,40.6,-73.9,40.8`
- Tokyo: `139.6,35.6,139.8,35.7`

### Choosing Between Territories and Bounding Box

**Use `--territories` when:**
- You want to sample from entire cities or regions
- You want automatic geographic boundary detection
- You need weighted sampling across multiple locations
- You want the system to handle geographic boundaries automatically

**Use `--bbox` when:**
- You need precise control over the sampling area
- You're analyzing a specific neighborhood or district
- You have exact coordinates from another mapping tool
- You want to avoid API calls to Nominatim for territory lookup
- You're doing comparative analysis across standardized areas

**Note:** You cannot use both `--territories` and `--bbox` in the same command. Choose one approach based on your analysis needs.

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