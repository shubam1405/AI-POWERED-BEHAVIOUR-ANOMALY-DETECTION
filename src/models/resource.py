from dataclasses import dataclass, field
from typing import List

from models.enums import (
    DepartmentName,
    ResourceType,
    SensitivityLevel,
)


@dataclass
class Resource:
    """
    Represents an enterprise resource that employees may access.
    """

    resource_name: ResourceType

    resource_type: str

    sensitivity_level: SensitivityLevel

    allowed_departments: List[DepartmentName] = field(default_factory=list)

    def is_access_allowed(self, department: DepartmentName) -> bool:
        """Returns True if the department is allowed to access this resource."""
        return department in self.allowed_departments

    def add_allowed_department(self, department: DepartmentName) -> None:
        """Grants access to a department."""
        if department not in self.allowed_departments:
            self.allowed_departments.append(department)

    def revoke_department(self, department: DepartmentName) -> None:
        """Revokes access from a department."""
        if department in self.allowed_departments:
            self.allowed_departments.remove(department)

    @property
    def is_restricted(self) -> bool:
        """Returns True if the resource is highly sensitive."""
        return self.sensitivity_level == SensitivityLevel.RESTRICTED

    def __str__(self) -> str:
        return f"{self.resource_name.value} ({self.resource_type})"