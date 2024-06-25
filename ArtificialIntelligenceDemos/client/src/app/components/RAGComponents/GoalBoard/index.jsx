
const GoalBoard = ({ boardData }) => {
    return (
        <div className="flex flex-col bg-gray-700 rounded p-4 border border-gray-600 w-1/2 m-auto mt-3">
            <h5 className="text-3xl text-center text-blue-400 mb-3"> Goal Board </h5>
            <table className="m-5 mt-0 h-full">
                {boardData.goal.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                        {row.map((number, index) => (
                            <td key={number + index} className="border p-2 text-center text-xl rounded">
                                {number}
                            </td>
                        ))}
                    </tr>
                ))}
            </table>
            <div className="flex gap-5 text-center">
                <span>
                    Depth {boardData.depth}
                </span>
                <span>
                    Euclidean {Number(boardData.euclideanValue).toFixed(2)}
                </span>
                <span>
                    Manhattan {Number(boardData.manhattanValue).toFixed(2)}
                </span>
                <span>
                    Path: [{boardData.path.map((step, index) =>
                        index != boardData.path.length - 1 ? step + ", " : step
                    )}]
                </span>
            </div>
        </div>
    )
}

export default GoalBoard;