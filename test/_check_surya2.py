import surya
import pkgutil

print('=== surya.layout ===')
from surya import layout
print(f'layout dir: {dir(layout)}')

print()
print('=== surya.models ===')
from surya import models
print(f'models dir: {dir(models)}')

print()
print('=== Checking LayoutPredictor ===')
try:
    from surya.layout import LayoutPredictor
    print('  LayoutPredictor found!')
except ImportError as e:
    print(f'  LayoutPredictor not found: {e}')

print()
print('=== surya.common ===')
from surya import common
print(f'common dir: {dir(common)}')

print()
print('=== surya.foundation ===')
from surya import foundation
print(f'foundation dir: {dir(foundation)}')

print()
print('=== Try importing surya_order (separate package) ===')
try:
    import surya_order
    print('  surya_order package found!')
    print(f'  dir: {dir(surya_order)}')
except ImportError as e:
    print(f'  surya_order not installed: {e}')
