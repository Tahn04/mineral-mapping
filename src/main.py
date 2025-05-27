import os


import core.tile_processing as tp

def main():

    # path = r"Mars_Data\T1250_demo_parameters"
    input_path = r"Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"
    output_path = r"Code\mineral-mapping\outputs"
    
    tp.TileParameterization(
        input_path,
        output_path,
        param="D2300"
    ).process()
    

#path = r"Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"

if __name__ == "__main__":
    main()