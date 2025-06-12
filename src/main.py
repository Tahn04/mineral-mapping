import core.tile_processing as tp
import core.vectroscopy as vp
import rasterio
import numpy as np


def main():
    """
    Main function to process a tile with specified parameters.
    """
    # path = r"Mars_Data\T1250_demo_parameters"
    # dir_path = r"Mars_Data\T1250_demo_parameters"
    # MC_path = r"Mars_Data\MC13_demo_parameters"
    # output_path = r"Code\mineral-mapping\outputs"

    # tp.TileParameterization(
    #     dir_path,
    #     output_path,
    #     name="D2300"
    # ).process_parameter()

    # tp.TileParameterization(
    #     dir_path,
    #     output_path,
    #     name="MonoSulfate"
    # ).process_indicator()

    vp.Vectroscopy.from_config().vectorize()
    
    path = r"\\lasp-store\home\taja6898\Documents\Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"

    with rasterio.open(path) as src:
        D2300 = src.read(1, masked=True).filled(np.nan)
        profile = src.profile  # get metadata
        transform = src.transform
        crs = src.crs

    # tp.ProcessingPipeline().process_file()

    vp.Vectroscopy.from_array(
        rast={
            "D2300": (D2300, [0.005, 0.0075, 0.01, 0.0125, 0.015])
        },
        mask=None,  # You can specify a mask if needed
        crs=crs,
        transform=transform
    ).vectorize(
        output="pandas"
    )


    # vectorize(
    #     rast={D2300: [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275]},
    #     mask=None,
    #     crs=crs,
    #     transform=transform,
    #     output="pandas"
    # )

    """
    vectorize(
        rast=[
            D2300,
            BD1900
        ],
        mask=BD1500,
        crs=crs,
        transform=transform
        output="pandas"
    )
    """
    

#path = r"Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"

if __name__ == "__main__":
    main()