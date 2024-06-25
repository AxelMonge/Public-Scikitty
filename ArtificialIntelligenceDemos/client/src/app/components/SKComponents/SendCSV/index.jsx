import { faSpinner } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

const CRITERIA = [
    "Entropy",
    "Gini"
]

const SendCSV = ({ fileName, values, onChangeTarget, onChangeCriterion, onEnviar, isLoading }) => {
    return (
        <section className="mt-3 flex flex-col">
            {fileName != "" && (
                <>
                    <p>
                        2. Please select the target you want to use to train the Decision Tree:
                    </p>
                    <div className="flex justify-center gap-5">
                        <span className="flex gap-3 justify-center my-5">
                            Target:
                            <select className="text-black rounded p-1" onChange={onChangeTarget} name="target">
                                {Object.keys(values).map((value, index) =>
                                    <option key={index}> {value} </option>
                                )}
                            </select>
                        </span>
                        <span className="flex gap-3 justify-center my-5">
                            Criterion:
                            <select className="text-black rounded p-1" onChange={onChangeCriterion} name="criterion">
                                {CRITERIA.map((value, index) => 
                                    <option key={index}> {value} </option>
                                )}
                            </select>
                        </span>
                    </div>
                </>
            )}
            {!isLoading ? (
                <button onClick={onEnviar} className={`w-1/3 m-auto mt-3 rounded p-2 ${fileName == "" ? "cursor-not-allowed bg-green-700 text-gray-400" : "bg-green-500"}`}>
                    {isLoading ? "" : "🚀 Send dataset to train a Decision Tree 🚀"}
                </button>
            ) : (
                <button
                    className={`w-1/3 m-auto mt-3 rounded p-2 cursor-not-allowed bg-green-700 text-gray-400"`}
                >
                    <FontAwesomeIcon icon={faSpinner} className="spinner" />
                </button>
            )}
        </section>
    )
}

export default SendCSV;