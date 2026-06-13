import surya
import pkgutil

print('surya version:', surya.__version__ if hasattr(surya, '__version__') else 'unknown')
print()
print('surya module contents:')
for importer, modname, ispkg in pkgutil.iter_modules(surya.__path__):
    print(f'  {modname} {"(package)" if ispkg else "(module)"}')

print()
print('Checking if surya.ordering exists...')
try:
    from surya import ordering
    print('  Yes, surya.ordering exists')
    print(f'  ordering contents: {dir(ordering)}')
except ImportError as e:
    print(f'  No: {e}')

print()
print('Checking if surya.inference exists...')
try:
    from surya import inference
    print('  Yes, surya.inference exists')
    print(f'  inference contents: {dir(inference)}')
except ImportError as e:
    print(f'  No: {e}')

print()
print('Checking surya.model...')
try:
    from surya import model
    print('  Yes, surya.model exists')
    print(f'  model submodules:')
    for importer, modname, ispkg in pkgutil.iter_modules(model.__path__):
        print(f'    {modname}')
except ImportError as e:
    print(f'  No: {e}')
