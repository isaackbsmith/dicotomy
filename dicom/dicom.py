# 2024, Isaac Smith

import pydicom
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Literal


class DICOM:
    """
    attributes:
        path (string): DICOM file/directory path
        dcm_files (Path[]): An array containing the dicom file(s) paths

    methods:
        process_static():
            Processes the DICOM pixel array, exports the files as images
            and returns an 8-bit representation.

        process_dynamic():
            Processes the DICOM pixel array, exports the files as images
            and returns an 8-bit representation. The contrast of the resulting
            image can be modified by passing in the min and max Hounsfield unit
            values.
    """

    def __init__(self, path: str) -> None:

        self.path = Path(path)
        self.dcm_files: list[Path] = []

        if not self.path.exists():
            raise FileNotFoundError("File/Folder does not exist")

        if self.path.is_file():
            self.dcm_files.append(self.path)
        else:
            self.dcm_files.extend(self.path.glob("*.dcm"))

    def process_static(self,
                       output_dir: str,
                       output_format: Literal["png", "jpg"] = "png",
                       plot: bool = False
                       ) -> list[np.uint8]:
        """
        Processes a DICOM pixel array by normalizing
        and converting to 8-bit unsigned integer format

        parameters:
            output_dir (string): Output directory 
            output_format (jpg or png): Image format to export to
            plot (boolean): Flag to plot image or not
        returns:
            np.uint8[]: Processed DICOM pixel arrays in the range [0, 255] as uint8
        """

        processed_files: list[np.uint8] = []
        output_path = Path(output_dir)

        if not output_path.exists():
            print("Provided path does not exist...creating path")
            output_path.mkdir(parents=True, exist_ok=True)

        for file in self.dcm_files:

            # Read DICOM file
            dcm_file = pydicom.dcmread(file)

            # Skip file if it has no pixel data
            if not hasattr(dcm_file, "pixel_array"):
                print("File has no pixel data...skipping")
                continue

            # Get pixel array
            pixel_array = dcm_file.pixel_array

            # Convert the values to floating-point for precise calculations
            pixel_array_f64 = pixel_array.astype(float)

            # Ensure all values are non-negative
            pixel_array_pos_f64 = np.maximum(pixel_array_f64, 0)

            # Normalize the values to a range of 0 to 1
            pixel_array_pos_norm_f64 = pixel_array_pos_f64 / pixel_array_pos_f64.max()

            # Scale the normalized values to a range of 0 to 255
            pixel_array_pos_norm_f64 *= 255.0

            # Convert the result to an 8-bit unsigned integer format
            pixel_array_pos_norm_u8 = np.uint8(pixel_array_pos_norm_f64)

            processed_files.append(pixel_array_pos_norm_u8)

            # Export file
            self._export(pixel_array_pos_norm_u8, file, output_path, output_format, plot)

        return processed_files


    def process_dynamic(self,
                        output_dir: str,
                        min_p: int | None = None,
                        max_p: int | None = None,
                        output_format: Literal["png", "jpg"] = "png",
                        plot: bool = False
                        ) -> list[np.uint8]:
        """
        Processes a DICOM pixel array by normalizing
        and converting to 8-bit unsigned integer format
        while allowing for the dynamic control of the
        minimum and maximum pixel intensities

        parameters:
            output_dir (string): Output directory 
            min_p (integer or nil): Minimum Hounsfield Unit
            max_p (integer or nil): Maximum Hounsfield Unit
            output_format (jpg or png): Image format to export to
            plot (boolean): Flag to plot image or not
        returns:
            np.uint8[]: Processed DICOM pixel arrays in the range [0, 255] as uint8
        """

        processed_files: list[np.uint8] = []
        output_path = Path(output_dir)

        if not output_path.exists():
            print("Provided path does not exist...creating path")
            output_path.mkdir(parents=True, exist_ok=True)

        for file in self.dcm_files:

            dcm_file = pydicom.dcmread(file)

            if not hasattr(dcm_file, "pixel_array"):
                print("File has no pixel data...skipping")
                continue

            pixel_array = dcm_file.pixel_array

            # Determine the minimum and maximum Hounsfield Unit pixel values
            if min_p: hounsfield_min = min_p
            hounsfield_min = np.min(pixel_array)

            if max_p: hounsfield_max = max_p
            hounsfield_max = np.max(pixel_array)

            # Find the range of the values in the pixel array
            hounsfield_range = hounsfield_max - hounsfield_min

            # Clip the pixel values to ensure they fall between the HU range
            pixel_array[pixel_array < hounsfield_min] = hounsfield_min
            pixel_array[pixel_array > hounsfield_max] = hounsfield_max

            # Normalize the pixel values to a range of 0 to 1
            pixel_array_norm = (pixel_array - hounsfield_min) / hounsfield_range

            pixel_array_norm *= 255

            pixel_array_norm_u8 = np.uint8(pixel_array_norm)

            processed_files.append(pixel_array_norm_u8)

            self._export(pixel_array_norm_u8, file, output_path, output_format, plot)

        return processed_files


    def _export(self,
               pixel_data_u8: np.uint8,
               file: Path,
               output_dir: Path,
               format: Literal["jpg", "png"],
               plot: bool = False) -> None:
        """
        Export unsigned 8-bit pixel array to JPEG or PNG image or a plot

        parameters:
            pixel_data_u8 (np.unint8): Pixel data in unsigned 8-bit format
            file (string): DICOM file
            output_dir (Path): Output directory
            format (jpg or png): Image format to export to
            plot (boolean): Flag to plot image or not

        returns:
            None
        """

        if plot:
            plt.imshow(pixel_data_u8, cmap=mpl.colormaps["gray"])
            Path(output_dir.joinpath("plt/")).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{output_dir}/plt/{file.stem}.{"jpg" if format == "jpg" else "png"}")
        else:
            image = Image.fromarray(pixel_data_u8)
            Path(output_dir.joinpath("img/")).mkdir(parents=True, exist_ok=True)
            image.save(f"{output_dir}/img/{file.stem}.{"jpg" if format == "jpg" else "png"}")


