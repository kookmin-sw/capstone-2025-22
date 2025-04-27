from produce_data.produce_dataset import ProduceDataset


if __name__ == "__main__":
    from configs import getconfig

    cofigpath = f"src/workspace/config.yaml"
    
    args = getconfig(cofigpath)

    handler = ProduceDataset(args)
    predict_result = handler.produce_all_dataset()
    # xml_tree = Annotation2Xml.annotation_to_musicxml(predict_result)

    # self.handler.predict(biImg_list)
