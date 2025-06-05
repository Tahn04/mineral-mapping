import core.tile_processing as tp


def main():
    """
    Main function to process a tile with specified parameters.
    """
    # path = r"Mars_Data\T1250_demo_parameters"
    dir_path = r"Mars_Data\T1250_demo_parameters"
    MC_path = r"Mars_Data\MC13_demo_parameters"
    output_path = r"Code\mineral-mapping\outputs"

    # tp.TileParameterization(
    #     dir_path,
    #     output_path,
    #     name="BD1900_2"
    # ).process_parameter()

    tp.TileParameterization(
        dir_path,
        output_path,
        name="MonoSulfate"
    ).process_indicator()

    # tp.ProcessingPipeline().process_file()

    

#path = r"Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"

if __name__ == "__main__":
    main()