from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

from models.enums import DepartmentName

if TYPE_CHECKING:
    from models.employee import Employee
    from models.resource import Resource


@dataclass
class Department:
    """
    Represents an organizational department within the enterprise.
    """

    department_id: str
    name: DepartmentName

    manager: Optional["Employee"] = None

    employees: List["Employee"] = field(default_factory=list)

    resources: List["Resource"] = field(default_factory=list)

    policies: dict[str, Any] = field(default_factory=dict)

    def add_employee(self, employee: "Employee") -> None:
        """Adds an employee to the department."""
        if employee not in self.employees:
            self.employees.append(employee)

    def remove_employee(self, employee: "Employee") -> None:
        """Removes an employee from the department."""
        if employee in self.employees:
            self.employees.remove(employee)

    def add_resource(self, resource: "Resource") -> None:
        """Assigns a resource to the department."""
        if resource not in self.resources:
            self.resources.append(resource)

    def remove_resource(self, resource: "Resource") -> None:
        """Removes a resource from the department."""
        if resource in self.resources:
            self.resources.remove(resource)

    def set_manager(self, manager: "Employee") -> None:
        """Assigns the department manager."""
        self.manager = manager

    def has_resource(self, resource: "Resource") -> bool:
        """Checks whether the department has access to a resource."""
        return resource in self.resources

    @property
    def employee_count(self) -> int:
        return len(self.employees)

    @property
    def resource_count(self) -> int:
        return len(self.resources)

    def __str__(self) -> str:
        return f"{self.department_id} ({self.name.value})"