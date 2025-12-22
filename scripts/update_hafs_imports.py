import os
import re

def get_shim_mappings():
    mappings = []
    # Check hafs shims (top-level)
    hafs_dir = "src/hafs"
    # List of all modularized packages
    modularized = [
        "config", "core", "models", "synergy", "context", 
        "backends", "agents", "services", "ui", "cli", 
        "llm", "adapters", "plugins", "integrations", "editors"
    ]
    
    for item in modularized:
        item_dir = os.path.join(hafs_dir, item)
        if os.path.exists(item_dir):
            for f in os.listdir(item_dir):
                if f.endswith(".py") and f != "__init__.py":
                    path = os.path.join(item_dir, f)
                    try:
                        with open(path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            # Match common patterns in shims
                            match = re.search(r'import ([^ ]+)', content)
                            if match:
                                module_name = f[:-3]
                                canonical = match.group(1).split()[0] # basic split
                                if canonical.startswith("tui.") or canonical == "tui":
                                    pass # will handle specifically if needed
                                
                                mappings.append((rf'\bhafs\.{item}\.{module_name}\b', canonical))
                    except Exception:
                        continue
    return mappings

patterns = [
    (r'\bhafs\.config\b', 'config'),
    (r'\bhafs\.core\b', 'core'),
    (r'\bhafs\.models\b', 'models'),
    (r'\bhafs\.synergy\b', 'synergy'),
    (r'\bhafs\.context\b', 'context'),
    (r'\bhafs\.backends\b', 'backends'),
    (r'\bhafs\.agents\b', 'agents'),
    (r'\bhafs\.services\b', 'services'),
    (r'\bcore\.services\b', 'services'),
    (r'\bhafs\.ui\b', 'tui'),
    (r'\bhafs\.cli\b', 'cli'),
    (r'\bhafs\.llm\b', 'llm'),
    (r'\bhafs\.adapters\b', 'adapters'),
    (r'\bhafs\.plugins\b', 'plugins'),
    (r'\bhafs\.integrations\b', 'integrations'),
    (r'\bhafs\.editors\b', 'editors'),
    (r'\bhafs\.data\b', 'data'),
]

# Add shim mappings
patterns.extend(get_shim_mappings())

# Sort by length of pattern (descending) to avoid partial matches
patterns.sort(key=lambda x: len(x[1]), reverse=True)

def update_imports(directory):
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py') or file.endswith('.tcss') or file == 'CMakeLists.txt':
                path = os.path.join(root, file)
                if os.path.basename(path) == 'update_hafs_imports.py':
                    continue
                    
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue
                
                new_content = content
                for pattern, replacement in patterns:
                    new_content = re.sub(pattern, replacement, new_content)
                
                if new_content != content:
                    print(f"Updating {path}")
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

if __name__ == "__main__":
    for d in ["src", "tests", "scripts", "docs"]:
        if os.path.exists(d):
            update_imports(d)
    for f in os.listdir("."):
        if f.endswith(".py") and f != "update_hafs_imports.py":
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                new_content = content
                for pattern, replacement in patterns:
                    new_content = re.sub(pattern, replacement, new_content)
                if new_content != content:
                    print(f"Updating {f}")
                    with open(f, 'w', encoding='utf-8') as file:
                        file.write(new_content)
            except UnicodeDecodeError:
                pass
