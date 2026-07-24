from config.config import RESOURCES_CONFIG
from models.resource import Resource
from simulator.company import Company


class ResourceGenerator:
    """
    Generates enterprise resources from configuration and registers
    them with the company.
    """

    def __init__(self, company: Company):
        self.company = company

    def generate(self) -> None:
        """Generate all configured resources."""

        for config in RESOURCES_CONFIG:

            resource = Resource(
                resource_name=config["name"],
                resource_type=config["type"],
                sensitivity_level=config["sensitivity"],
                allowed_departments=list(config["allowed_departments"]),
            )

            self.company.add_resource(resource)