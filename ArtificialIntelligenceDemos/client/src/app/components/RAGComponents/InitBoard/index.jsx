"use client";
import { useState, useEffect } from "react";

const InitBoard = ({ boardSize, title, identifier }) => {
    const columns = Array.from({ length: boardSize }, (_, i) => i);
    const [formBoard, setFormBoard] = useState([]);
    const [randomBoard, setRandomBoard] = useState([]);

    useEffect(() => {
        if (randomBoard.length) {
            setFormBoard(randomBoard);
        }
    }, [randomBoard]);

    const randomizeBoardNumbers = () => {
        const totalNumbers = boardSize * boardSize;
        const numbers = Array.from({ length: totalNumbers }, (_, index) => index);

        for (let i = totalNumbers - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [numbers[i], numbers[j]] = [numbers[j], numbers[i]];
        }

        const matrix = [];
        for (let i = 0; i < boardSize; i++) {
            matrix.push(numbers.slice(i * boardSize, (i + 1) * boardSize));
        }

        setRandomBoard(matrix);
    }

    return (
        <div
            className="flex flex-col justify-center bg-gray-700 p-5 rounde w-1/2 h-full m-auto rounded border border-gray-600"
        >
            <h5 className="text-3xl text-center text-blue-400 mb-3">
                {title}
            </h5>
            <table className="m-5 mt-0 h-full">
                <tbody>
                    {columns.map((col1, index1) => (
                        <tr key={col1 + index1}>
                            {columns.map((col2, index2) => (
                                <td key={col2 + index2}>
                                    <input
                                        type="number"
                                        min={0}
                                        max={(boardSize * boardSize) - 1}
                                        className="border p-2 text-center text-xl text-black rounded w-full"
                                        name={`${identifier}${col1}${col2}`}
                                        defaultValue={formBoard[col1]?.[col2] ?? ''}
                                    />
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
            <button
                className="text-blue-400 bg-gray-800 p-4 rounded border border-gray-600 hover:bg-gray-900"
                onClick={randomizeBoardNumbers}
                type="button"
            >
                🎲 Randomize Board Numbers 🎲
            </button>
        </div>
    )
}

export default InitBoard;
