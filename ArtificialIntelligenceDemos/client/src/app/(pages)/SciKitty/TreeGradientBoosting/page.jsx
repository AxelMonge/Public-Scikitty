"use client";
import TopicHeader from "@/app/components/Topic/TopicHeader";
import Papa from "papaparse";
import UploadCSV from "@/app/components/SKComponents/UploadCSV";
import SendCSV from "@/app/components/SKComponents/SendCSV";
import Metrics from "@/app/components/SKComponents/Metrics";
import DoQuestion from "@/app/components/SKComponents/DoQuestion";
import { useState } from "react";
import { toast } from "sonner";
import { DATASETS } from "@/app/constants/DATASETS";

const DESC = `
    Tree Gradient Boosting is an ensemble learning technique that combines multiple 
    decision trees to improve predictive performance. It is constructed using a version 
    of a decision tree with a height of 1, without numerizing or encoding categorical variables. 
    The training process involves building trees sequentially, where each tree corrects 
    the errors of the previous one.
`

const TreeGradientBoosting = ({ }) => {
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
        const response = await fetch('http://127.0.0.1:8000/api/scikitty_tree_boosting/', {
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
        <main>
            <article className="text-[#ffffff] pb-10">
                <TopicHeader title="SciKitty Tree Boosting" description={DESC} />
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
                            <Metrics metrics={metrics} title={"Scikitty"} showIsBalanced={false} />
                            <Metrics metrics={skMetrics} title={"Sklearn"} showIsBalanced={false} />
                        </>
                    )}
                </section>
            </article>
        </main>
    )
}

export default TreeGradientBoosting;
