# Zelda-Themed Model Names for hafs_scawful

Naming scheme for fine-tuned models following Zelda universe.

## Naming Format

`<item/character>-<size>-<date>-<tier>`

Examples:
- `master-sword-1.5b-20251221-gold`
- `triforce-7b-20251221-silver`
- `hookshot-1.5b-20251221-alpha`

## Quality Tiers

| Tier | Quality Range | Zelda Reference |
|------|---------------|-----------------|
| **gold** | >= 0.6 avg | Triforce complete |
| **silver** | 0.5-0.6 | Silver arrows |
| **bronze** | 0.4-0.5 | Bronze gauntlets |
| **alpha** | Experimental | Alpha testing |
| **beta** | Testing | Beta testing |

## Suggested Names by Category

### Legendary Items (High Quality / Production)
- `master-sword` - The legendary blade
- `triforce` - Ultimate power
- `light-arrows` - Sacred weapon
- `hylian-shield` - Ultimate defense
- `ocarina` - Ocarina of Time

### Regular Items (Mid Quality)
- `hookshot` - Versatile tool
- `boomerang` - Reliable tool
- `bow` - Classic weapon
- `bombs` - Explosive power
- `mirror-shield` - Reflective defense

### Characters (Special Builds)
- `link` - The hero
- `zelda` - The princess
- `ganon` - The villain
- `navi` - The guide
- `epona` - Loyal steed

### Locations (Domain-Specific)
- `hyrule` - Overworld knowledge
- `kakariko` - Village data
- `death-mountain` - Dungeon specialist
- `lost-woods` - Exploration
- `temple-time` - Historical data

### Experimental (Alpha/Beta)
- `deku-stick` - Quick and dirty
- `slingshot` - Early attempt
- `bottle` - Flexible use
- `rupee` - Value testing

## Current Training

**Dataset:** 504 samples, 0.50 avg quality
**Suggested Name:** `master-sword-1.5b-20251221-gold`

Quality is solid (0.50), dataset covers multiple domains (asm, gigaleak, oracle, yaze), ready for production use.

## Usage

```bash
# Train with Zelda name
./scripts/train_windows.sh master-sword

# Or specify full name
./scripts/train_windows.sh master-sword-1.5b gold
```

## Future Naming

As you train more models:
- `triforce-wisdom-7b` - Large model for complex tasks
- `triforce-power-14b` - Massive model
- `triforce-courage-1.5b` - Fast model for quick tasks

Collect all three Triforce pieces for complete coverage!
