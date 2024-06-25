"use client";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useEffect, useState } from "react";
import { toast } from "sonner";

const TopicForm = ({ values, questionTarget, fileName }) => {
    const [answer, setAnswer] = useState("");
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setAnswer("");
    }, [questionTarget]);

    const onSubmit = async (e) => {
        try {
            e.preventDefault();
            setLoading(true);
            const form = e.target;
            const question = [];

            Object.keys(values).map((key) => {
                if (key != questionTarget) {
                    const selector = form[key.toLowerCase()];
                    question.push(selector.value);
                }
            });

            const response = await fetch('/api/question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ file_name: fileName, pregunta: question }),
            });

            if (!response.ok) {
                throw new Error(`Server Error: ${response.status} ${response.statusText}`);
            }
            else {
                const result = await response.json();
                setAnswer(result.respuesta);
            };
        }
        catch (e) {
            if (e instanceof TypeError) {
                toast.error('Error!', { description: "Server not online!" });
            }
            else {
                toast.error('Error!', { description: e.message });
            };
        }
        finally {
            setLoading(false);
        };
    };

    return (
        <form className="flex flex-col gap-5 w-1/2 m-auto" onSubmit={onSubmit}>
            <div className="flex flex-col gap-5">
                <div className="flex flex-col gap-5 max-h-[45vh] overflow-auto">
                    {Object.keys(values).map((key, index) =>
                        key !== questionTarget && (
                            <span key={index} className="flex gap-5" >
                                <span className="w-1/3"> {key} </span>
                                <select className="w-full text-black" name={key.toLowerCase()}>
                                    {values[key].sort().map((option, index) =>
                                        <option key={index} value={option}> {option} </option>
                                    )}
                                </select>
                            </span>
                        )
                    )}
                </div>
                {!loading ? (
                    <button className="bg-green-500 rounded p-1" type="submit">
                        🚀 Predict 🚀
                    </button>
                ) : (
                    <button
                        className={`w-full rounded p-1 cursor-not-allowed bg-green-700 text-gray-400"`}
                    >
                        <FontAwesomeIcon icon={faSpinner} className="spinner" />
                    </button>
                )}
            </div>
            {answer && (
                <h5 className="text-xl text-center">
                    {questionTarget + ": " + answer}
                </h5>
            )}
        </form>
    )
}

export default TopicForm;