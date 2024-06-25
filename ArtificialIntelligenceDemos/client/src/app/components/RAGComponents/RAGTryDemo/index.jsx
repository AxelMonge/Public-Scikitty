"use client";
import GoalBoard from "../GoalBoard";
import { useState } from "react";
import FormBoards from "../FormBoards";

const RAGTryDemo = ({ }) => {
    const [boardData, setBoardData] = useState({});
    const [boardSize, setBoardSize] = useState(2);

    const onChangeSize = (e) => {
        const newSize = e.target.value;
        if (newSize < 2 || newSize > 10) return;
        setBoardSize(newSize);
    }

    return (
        <>
            <p>
                Try to set a size for the board. Insert the numbers in
                the board o try to use the Randomize Board Numbers Button.
                Use '0' for the empty value.
            </p>
            <div className="flex justify-center items-center gap-5 mt-3 ">
                <span className="bg-gray-800 p-3 flex justify-center items-center w-1/3 rounded gap-5 border border-gray-600">
                    <h5 className="text-blue-400"> 📐 Board Size </h5>
                    <input
                        className="text-black text-center rounded p-1"
                        type="number"
                        min={2}
                        onChange={onChangeSize}
                        defaultValue={boardSize}
                    />
                </span>
            </div>
            <FormBoards boardSize={boardSize} setBoardData={setBoardData} />
            {Object.keys(boardData).length > 0 && (
                <GoalBoard boardData={boardData} />
            )}
        </>
    )
}

export default RAGTryDemo;