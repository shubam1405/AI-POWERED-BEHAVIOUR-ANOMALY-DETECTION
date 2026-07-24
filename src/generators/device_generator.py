import random
import uuid
from typing import List

from config.config import (
    DEPARTMENT_DEVICE_TYPE_DISTRIBUTION,
    DEPARTMENT_OS_DISTRIBUTION,
    OS_BROWSER_DISTRIBUTION,
)
from models.device import Device
from models.employee import Employee
from models.enums import (
    Browser,
    DepartmentName,
    DeviceType,
    OperatingSystem,
)
from simulator.company import Company


class DeviceGenerator:
    """
    Generates devices for employees based on departmental policies.
    """

    def __init__(self, company: Company, rng: random.Random):
        self.company = company
        self.rng = rng
        self.device_counter = 0

    def _generate_device_id(self) -> str:
        self.device_counter += 1
        return f"DEV-{self.device_counter:04d}"

    def _choose_from_distribution(self, distribution: dict):
        items = list(distribution.keys())
        weights = list(distribution.values())
        return self.rng.choices(items, weights=weights, k=1)[0]

    def _generate_uuid(self) -> str:
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def _generate_mobile_device(self):
        os = self.rng.choices(
            [OperatingSystem.IOS, OperatingSystem.ANDROID],
            weights=[0.5, 0.5],
            k=1,
        )[0]

        browser = self._choose_from_distribution(
            OS_BROWSER_DISTRIBUTION.get(
                os,
                {Browser.CHROME: 1.0},
            )
        )

        return DeviceType.MOBILE, os, browser

    def generate_for_employee(self, employee: Employee) -> List[Device]:
        """
        Generates between one and three devices for an employee.
        """

        department = employee.department

        if department == DepartmentName.SALES:
            num_devices = self.rng.choices(
                [1, 2, 3],
                weights=[0.20, 0.60, 0.20],
                k=1,
            )[0]

        elif department == DepartmentName.IT:
            num_devices = self.rng.choices(
                [1, 2, 3],
                weights=[0.30, 0.40, 0.30],
                k=1,
            )[0]

        else:
            num_devices = self.rng.choices(
                [1, 2, 3],
                weights=[0.60, 0.30, 0.10],
                k=1,
            )[0]

        devices: List[Device] = []

        for index in range(num_devices):

            if index == 0:
                device_type = self._choose_from_distribution(
                    DEPARTMENT_DEVICE_TYPE_DISTRIBUTION.get(
                        department,
                        {DeviceType.LAPTOP: 1.0},
                    )
                )

                operating_system = self._choose_from_distribution(
                    DEPARTMENT_OS_DISTRIBUTION.get(
                        department,
                        {OperatingSystem.WINDOWS: 1.0},
                    )
                )

                browser = self._choose_from_distribution(
                    OS_BROWSER_DISTRIBUTION.get(
                        operating_system,
                        {Browser.CHROME: 1.0},
                    )
                )

            else:
                if department == DepartmentName.SALES and index == 1:
                    (
                        device_type,
                        operating_system,
                        browser,
                    ) = self._generate_mobile_device()

                else:
                    device_type = self.rng.choices(
                        [DeviceType.LAPTOP, DeviceType.MOBILE],
                        weights=[0.70, 0.30],
                        k=1,
                    )[0]

                    if device_type == DeviceType.MOBILE:
                        (
                            device_type,
                            operating_system,
                            browser,
                        ) = self._generate_mobile_device()

                    else:
                        operating_system = self._choose_from_distribution(
                            DEPARTMENT_OS_DISTRIBUTION.get(
                                department,
                                {OperatingSystem.WINDOWS: 1.0},
                            )
                        )

                        browser = self._choose_from_distribution(
                            OS_BROWSER_DISTRIBUTION.get(
                                operating_system,
                                {Browser.CHROME: 1.0},
                            )
                        )

            device = Device(
                device_uuid=self._generate_uuid(),
                device_id=self._generate_device_id(),
                device_type=device_type,
                operating_system=operating_system,
                browser=browser,
                owner_id=employee.employee_id,
            )

            self.company.add_device(device)
            employee.add_device(device.device_id)
            devices.append(device)

        return devices