"use client";
import TopicHeader from "@/app/components/Topic/TopicHeader";
import Papa from "papaparse";
import UploadCSV from "@/app/components/SKComponents/UploadCSV";
import Image from "next/image";
import { useState } from "react";
import { DATASETS } from "@/app/constants/DATASETS";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";
import { toast } from "sonner";

const DESC = `
    Linear Regression is a statistical method used to model the relationship between 
    a dependent variable and one or more independent variables. The preferred dataset 
    for testing is California Housing because it represents a problem that can be decomposed 
    into a linear trend. It predicts outcomes by fitting a linear equation to the observed data.
`

const LinearRegression = ({ }) => {
    const [result, setResult] = useState(null);
    const [csvData, setCSVData] = useState([]);
    const [values, setValues] = useState({});
    const [questionTarget, setQuestionTarget] = useState("");
    const [fileName, setFileName] = useState("");
    const [dataSets, setDataSets] = useState(DATASETS);
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

    const handleOnSelectSavedDataSet = (e) => {
        e.preventDefault();
        const dataSetIndex = e.target.value;
        if (dataSetIndex == 0) return;
        const newDataSet = dataSets[dataSetIndex - 1];
        setFileName(newDataSet.fileName);
        setCSVData(newDataSet.data);
        initParams(newDataSet);
    };

    const getImage = async () => {
        try {
            if (fileName == "") return;
            setLoading(true);
            const response = await fetch('http://127.0.0.1:8000/api/scikitty_linear_regression/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    csv: csvData,
                    featureTarget: questionTarget,
                    fileName: fileName,
                }),
            });

            if (response.ok) {
                const result = await response.json();
                setResult(result);
            } else {
                throw new Error(`Server Error: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            if (e instanceof TypeError) {
                toast.error('Error!', { description: "Server not online!" });
            }
            else {
                toast.error('Error!', { description: e.message });
            }
        }
        finally {
            setLoading(false);
        }
    }

    const getCaliforniaHousing = async () => {
        try {
            setLoading(true);
            const response = await fetch('http://127.0.0.1:8000/api/scikitty_california_housing/', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.ok) {
                const result = await response.json();
                setResult(result);
            } else {
                throw new Error(`Server Error: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            if (e instanceof TypeError) {
                toast.error('Error!', { description: "Server not online!" });
            }
            else {
                toast.error('Error!', { description: e.message });
            }
        }
        finally {
            setLoading(false);
        }
    }

    return (
        <main>
            <article className="text-[#ffffff] pb-10">
                <TopicHeader title="SciKitty Linear Regression" description={DESC} />
                <section className="mx-10 mt-10">
                    <h2 className="text-3xl"> Try Demo </h2>
                    <hr className="my-3" />
                    <div>
                        <UploadCSV
                            changeHandler={changeHandler}
                            dataSets={dataSets}
                            onSelectSavedDataSet={handleOnSelectSavedDataSet}
                        />
                        {fileName != "" && (
                            <div className="mt-3">
                                <p>
                                    2. Please select the target you want to use to train the Decision Tree:
                                </p>
                                <div className="flex justify-center gap-5">
                                    <span className="flex gap-3 justify-center my-5">
                                        Target:
                                        <select className="text-black rounded p-1" onChange={handleOnChangeTarget} name="target">
                                            {Object.keys(values).map((value, index) =>
                                                <option key={index}> {value} </option>
                                            )}
                                        </select>
                                    </span>
                                </div>
                            </div>
                        )}
                        {!loading ? (
                            <div className="flex justify-center gap-5">
                                <button onClick={getImage} className="bg-green-500 w-1/3 rounded p-2 mt-3 hover:bg-green-700">
                                    🚀 Send Dataset to Create the Linear Regression 🚀
                                </button>
                                <button onClick={getCaliforniaHousing} className="bg-green-500 w-1/3 rounded p-2 mt-3 hover:bg-green-700">
                                    🚀 Try with Sklearn California Housing Dataset 🚀
                                </button>
                            </div>
                        ) : (
                            <div className="flex justify-center">
                                <button
                                    className={`w-1/3 m-auto mt-3 rounded p-2 cursor-not-allowed bg-green-700 text-gray-400"`}
                                >
                                    <FontAwesomeIcon icon={faSpinner} className="spinner" />
                                </button>
                            </div>
                        )}
                        {result && (
                            <div className="bg-gray-800 py-10 rounded flex flex-col items-center mt-5 gap-8 border border-gray-600">
                                <span className="text-center">
                                    <h5 className="text-blue-400 text-xl"> Mean Squared Error </h5>
                                    {result.mse}
                                </span>
                                <Image
                                    width={3200}
                                    height={1600}
                                    src={`data:image/png;base64,${result.plot}`}
                                    alt="Plot"
                                    className="w-2/3"
                                />
                            </div>
                        )}
                    </div>
                </section>
            </article>
        </main>
    )
}

export default LinearRegression;