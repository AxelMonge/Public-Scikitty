
const UploadCSV = ({ changeHandler, dataSets, onSelectSavedDataSet }) => {
    return (
        <section className="flex flex-col">
            <p className="mb-3">
                1. First, it is necessary to select and upload a dataset in CSV format
                or choose a default dataset from the selector.
            </p>
            <div className="flex m-auto gap-10 items-center bg-gray-800 rounded p-7">
                <input
                    type="file"
                    name="file"
                    accept=".csv"
                    onChange={changeHandler}
                />
                <select className="rounded bg-gray-700 p-5" onChange={onSelectSavedDataSet} name="dataSet">
                    <option value={0}> Choose a saved dataset </option>
                    {dataSets.map((dataset, index) => (
                        <option key={index + 1} value={index + 1}> {dataset.fileName} </option>
                    ))}
                </select>
            </div>
        </section>
    )
}

export default UploadCSV;