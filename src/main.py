import core.tile_processing as tp


def main():
    """
    Main function to process a tile with specified parameters.
    """
    # path = r"Mars_Data\T1250_demo_parameters"
    dir_path = r"Mars_Data\T1250_demo_parameters"
    output_path = r"Code\mineral-mapping\outputs"

    # tp.TileParameterization(
    #     dir_path,
    #     output_path,
    #     name="D2300"
    # ).process_parameter()

    # tp.TileParameterization(
    #     dir_path,
    #     output_path,
    #     name="PolySulfate"
    # ).process_indicator()

    tp.ProcessingPipeline()

    

#path = r"Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"

if __name__ == "__main__":
    main()