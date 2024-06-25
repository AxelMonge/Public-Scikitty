"use client";
import Papa from "papaparse";
import UploadCSV from "../UploadCSV";
import SendCSV from "../SendCSV";
import Metrics from "../Metrics";
import DoQuestion from "../DoQuestion";
import { useState } from "react";
import { toast } from "sonner";
import { DATASETS } from "@/app/constants/DATASETS";
import Image from "next/image";

const SKTryDemo = ({ }) => {
    const [csvData, setCSVData] = useState([]);
    const [values, setValues] = useState({});
    const [questionTarget, setQuestionTarget] = useState("");
    const [criterion, setCriterion] = useState("entropy");
    const [fileName, setFileName] = useState("");
    const [dataSets, setDataSets] = useState(DATASETS);
    const [metrics, setMetrics] = useState([]);
    const [skMetrics, setSkMetrics] = useState([]);
    const [loading, setLoading] = useState(false);

    const changeHandler = (e) => {
        const form = e.target;
        if (!form.files[0]) return;
        const fileName = form.files[0].name;
        const nameWithoutExtension = fileName.substring(0, fileName.lastIndexOf("."));
        const newFileName = nameWithoutExtension;
        const newDataSet = {
            fileName: newFileName,
        }
        setFileName(newFileName);
        Papa.parse(form.files[0], {
            header: false,
            skipEmptyLines: true,
            complete: function (results) {
                setCSVData(results.data);
                newDataSet.data = results.data;
                initParams(newDataSet);
            },
        });
        setDataSets([...dataSets, newDataSet]);
    };

    const initParams = (newDataSet) => {
        if(newDataSet.data.length == 0) return
        const header = newDataSet.data[0];
        const dataWithOutHeader = newDataSet.data.slice(1);
        setQuestionTarget(header[0]);
        const output = dataWithOutHeader[0].map((_, colIndex) => dataWithOutHeader.map(row => row[colIndex]));
        let newDic = {};
        output.map((row, index) => {
            const a = ([... new Set(row)]);
            newDic = {
                ...newDic,
                [header[index]]: a,
            }
        });
        setValues(newDic);
    };

    const handleOnChangeTarget = (e) => {
        e.preventDefault();
        const newOption = e.target.value;
        setQuestionTarget(newOption);
    };

    const handleOnChangeCriterion = (e) => {
        e.preventDefault();
        const newOption = e.target.value;
        const lowerCase = newOption.toLowerCase();
        setCriterion(lowerCase);
    };

    const handleOnSelectSavedDataSet = (e) => {
        e.preventDefault();
        const dataSetIndex = e.target.value;
        if (dataSetIndex == 0) return;
        const newDataSet = dataSets[dataSetIndex - 1];
        setFileName(newDataSet.fileName);
        setCSVData(newDataSet.data);
        initParams(newDataSet);
    };

    const onEnviar = async () => {
        try {
            if (fileName == "") return;
            setLoading(true);
            const scikittyDT = await getScikittyDT();
            const sklearnDT = await getSklearnDT();
            setMetrics(scikittyDT);
            setSkMetrics(sklearnDT);
        }
        catch (e) {
            if (e instanceof TypeError) {
                toast.error('Error!', { description: "Server not online!" });
            }
            else {
                toast.error('Error!', { description: e.message });
            }
        }
        finally {
            setLoading(false);
        };
    };

    const getScikittyDT = async () => {
        const response = await fetch('http://127.0.0.1:8000/api/transformCSV/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                csv: csvData,
                featureTarget: questionTarget,
                fileName: fileName,
                criterion: criterion,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status} ${response.statusText}`);
        }
        else {
            const result = await response.json();
            return result;
        };
    };

    const getSklearnDT = async () => {
        const response = await fetch('http://127.0.0.1:8000/api/sklearnDT/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                csv: csvData,
                featureTarget: questionTarget,
                fileName: fileName,
                criterion: criterion,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status} ${response.statusText}`);
        }
        else {
            const result = await response.json();
            return result;
        };
    };

    return (
        <section className="mx-10 mt-10">
            <h2 className="text-3xl"> Try Demo </h2>
            <hr className="my-3" />
            <UploadCSV
                changeHandler={changeHandler}
                dataSets={dataSets}
                onSelectSavedDataSet={handleOnSelectSavedDataSet}
            />
            <SendCSV
                fileName={fileName}
                values={values}
                onChangeTarget={handleOnChangeTarget}
                onChangeCriterion={handleOnChangeCriterion}
                onEnviar={onEnviar}
                isLoading={loading}
            />
            {Object.keys(metrics).length > 0 && Object.keys(skMetrics).length > 0 && (
                <>
                    <Metrics metrics={metrics} title={"Scikitty"} showIsBalanced={true} />
                    <Metrics metrics={skMetrics} title={"Sklearn"} showIsBalanced={false} />
                    <div className="bg-gray-800 mt-3 p-10 rounded border border-gray-700 flex justify-around gap-5">
                        <Image
                            width={3200}
                            height={1600}
                            src={`data:image/png;base64,${metrics.plot_base64}`}
                            alt="Plot"
                            className="w-1/3"
                        />
                        <Image
                            width={3200}
                            height={1600}
                            src={`data:image/png;base64,${skMetrics.plot_base64}`}
                            alt="Plot"
                            className="w-2/3"
                        />
                    </div>
                    <DoQuestion
                        values={values}
                        questionTarget={questionTarget}
                        fileName={fileName}
                    />
                </>
            )}
        </section>
    )
}

export default SKTryDemo;