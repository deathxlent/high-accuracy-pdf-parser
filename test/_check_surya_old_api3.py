import sys
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

from surya.schema import OrderBox
print("=== OrderBox fields ===")
for field_name, field_info in OrderBox.model_fields.items():
    print(f"  {field_name}: {field_info.annotation}")

from surya.ordering import OrderBox as OB2
print("\n=== surya.ordering.OrderBox ===")
for field_name, field_info in OB2.model_fields.items():
    print(f"  {field_name}: {field_info.annotation}")
