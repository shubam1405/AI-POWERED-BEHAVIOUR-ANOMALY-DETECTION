from typing import Dict, List, Optional
from models.enums import DepartmentName
from models.employee import Employee
from models.device import Device
from models.resource import Resource
from models.department import Department

class Company:
    """In-memory representation of the virtual enterprise (source of truth)."""
    def __init__(self, name: str = "Virtual Enterprise"):
        self.name: str = name
        self.departments: Dict[DepartmentName, Department] = {}
        self.employees: Dict[str, Employee] = {}  # key: employee_id (readable)
        self.devices: Dict[str, Device] = {}      # key: device_id (readable)
        self.resources: Dict[str, Resource] = {}  # key: resource_name

    def add_department(self, department: Department) -> None:
        """Adds a department to the company."""
        self.departments[department.name] = department

    def add_employee(self, employee: Employee) -> None:
        """Adds an employee and maps them to their department."""
        self.employees[employee.employee_id] = employee
        # Bind employee to the Department object
        dep = self.departments.get(employee.department)
        if dep:
            if employee not in dep.employees:
                dep.employees.append(employee)
            # Assign department manager if missing or matching role
            if "Director" in employee.role or "Manager" in employee.role:
                if dep.manager is None:
                    dep.manager = employee

    def add_device(self, device: Device) -> None:
        """Registers a device with the company and links it to its owner."""
        self.devices[device.device_id] = device
        
        # Link the device to the employee's allowed list
        emp = self.get_employee(device.owner_id)
        if emp:
            if device.device_id not in emp.allowed_devices:
                emp.allowed_devices.append(device.device_id)

    def add_resource(self, resource: Resource) -> None:
        """Registers a resource and lists it under allowed departments."""
        self.resources[resource.resource_name] = resource
        for dep_name in resource.allowed_departments:
            dep = self.departments.get(dep_name)
            if dep:
                if resource not in dep.resources:
                    dep.resources.append(resource)

    def get_employee(self, employee_id: str) -> Optional[Employee]:
        """Look up an employee by either their readable employee_id (e.g. EMP-0043) or UUID."""
        if employee_id in self.employees:
            return self.employees[employee_id]
        
        # Fallback to searching by UUID
        for emp in self.employees.values():
            if emp.employee_uuid == employee_id:
                return emp
        return None

    def get_devices(self, employee_id: Optional[str] = None) -> List[Device]:
        """Retrieves all devices, or devices belonging to a specific employee ID/UUID."""
        if employee_id:
            emp = self.get_employee(employee_id)
            if emp:
                return [self.devices[d_id] for d_id in emp.allowed_devices if d_id in self.devices]
            return []
        return list(self.devices.values())

    def get_resources(self) -> List[Resource]:
        """Retrieves all registered resources."""
        return list(self.resources.values())

    def get_department_policy(self, department_name: DepartmentName) -> dict:
        """Retrieves policy configuration for a given department name."""
        dep = self.departments.get(department_name)
        return dep.policies if dep else {}
